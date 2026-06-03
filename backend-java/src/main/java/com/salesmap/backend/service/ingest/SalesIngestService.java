package com.salesmap.backend.service.ingest;

import com.salesmap.backend.dto.IndustryCategory;
import com.salesmap.backend.repository.SalesRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/** Facade — fetch + adapt + aggregate + upsert. */
@Service
public class SalesIngestService {

    private static final Logger log = LoggerFactory.getLogger(SalesIngestService.class);

    private final SalesRepository repo;
    private final OpenApiClient client;
    private final Path industryMapPath;

    public SalesIngestService(
        SalesRepository repo,
        OpenApiClient client,
        @Value("${salesmap.industry-map-path:}") String industryMapPathProp
    ) {
        this.repo = repo;
        this.client = client;
        this.industryMapPath = industryMapPathProp == null || industryMapPathProp.isBlank()
            ? defaultIndustryMapPath() : Paths.get(industryMapPathProp);
    }

    private static Path defaultIndustryMapPath() {
        // 작업 디렉토리 기준 infra/db/industry_map.csv
        return Paths.get("infra", "db", "industry_map.csv").toAbsolutePath();
    }

    public IngestResult run(
        List<String> quarters,
        List<Integer> regionIds,
        List<IndustryCategory> industries,
        String source,
        Iterator<Map<String, Object>> directRows
    ) {
        IndustryMap industryMap = IndustryMap.fromCsv(industryMapPath);
        RegionResolver regionResolver = new RegionResolver(repo);
        OpenApiRowAdapter adapter = new OpenApiRowAdapter(industryMap, regionResolver);

        Iterator<Map<String, Object>> rows = directRows;
        if (rows == null) {
            List<String> qs = (quarters == null) ? Collections.singletonList(null) :
                quarters.stream().map(SalesIngestService::toApiQuarter).toList();
            rows = chainFetch(qs);
        }

        int processed = 0, accepted = 0, failed = 0, deduped = 0, negatives = 0;
        List<Map<String, String>> errors = new ArrayList<>();
        List<NormalizedRow> normalized = new ArrayList<>();
        Set<Long> seenRaw = new HashSet<>();

        while (rows.hasNext()) {
            Map<String, Object> raw = rows.next();
            processed++;
            long fp = rawFingerprint(raw);
            if (!seenRaw.add(fp)) { deduped++; continue; }
            AdaptResult result = adapter.adapt(raw);
            if (!result.isOk()) {
                failed++;
                String reason = result.reason();
                if (reason.startsWith("NEGATIVE_")) negatives++;
                if (errors.size() < 50) errors.add(Map.of("reason", reason));
                continue;
            }
            NormalizedRow row = result.row();
            if (regionIds != null && !regionIds.isEmpty() && !regionIds.contains(row.regionId())) continue;
            if (industries != null && !industries.isEmpty() && !industries.contains(row.industry())) continue;
            normalized.add(row);
            accepted++;
        }

        List<SalesRepository.SalesRow> aggregated = aggregate(normalized);
        int upserted = repo.upsertMany(aggregated);

        TreeSet<String> obsQuarters = new TreeSet<>();
        TreeSet<String> obsIndustries = new TreeSet<>();
        for (NormalizedRow r : normalized) {
            obsQuarters.add(r.quarter());
            obsIndustries.add(r.industry().name());
        }

        log.info("ingest_done source={} processed={} accepted={} failed={} deduped={} negatives={} upserted={} quarters={}",
            source, processed, accepted, failed, deduped, negatives, upserted, obsQuarters);

        IngestResult out = new IngestResult();
        out.source = source;
        out.quarters = obsQuarters.isEmpty() ? (quarters != null ? quarters : List.of()) : new ArrayList<>(obsQuarters);
        out.industries = obsIndustries.isEmpty()
            ? (industries != null ? industries.stream().map(Enum::name).toList() : List.of())
            : new ArrayList<>(obsIndustries);
        out.processedRows = processed;
        out.acceptedRows = accepted;
        out.upsertedRows = upserted;
        out.failedRows = failed;
        out.dedupedRows = deduped;
        out.negativeRows = negatives;
        out.errors = errors;
        return out;
    }

    private Iterator<Map<String, Object>> chainFetch(List<String> yqs) {
        return new Iterator<>() {
            int idx = 0;
            Iterator<Map<String, Object>> cur = null;
            @Override
            public boolean hasNext() {
                while ((cur == null || !cur.hasNext()) && idx < yqs.size()) {
                    cur = client.fetchQuarter(yqs.get(idx++));
                }
                return cur != null && cur.hasNext();
            }
            @Override
            public Map<String, Object> next() { return cur.next(); }
        };
    }

    private static List<SalesRepository.SalesRow> aggregate(List<NormalizedRow> rows) {
        Map<List<Object>, long[]> agg = new LinkedHashMap<>();
        for (NormalizedRow r : rows) {
            List<Object> key = List.of(r.regionId(), r.quarter(), r.industry().name());
            long[] slot = agg.computeIfAbsent(key, k -> new long[]{0, 0});
            slot[0] += r.totalSales();
            slot[1] += r.totalCount() == null ? 0 : r.totalCount();
        }
        List<SalesRepository.SalesRow> out = new ArrayList<>(agg.size());
        for (var e : agg.entrySet()) {
            int regionId = (int) e.getKey().get(0);
            String quarter = (String) e.getKey().get(1);
            String industry = (String) e.getKey().get(2);
            long[] v = e.getValue();
            out.add(new SalesRepository.SalesRow(regionId, quarter, industry, v[0], v[1] == 0 ? null : v[1]));
        }
        return out;
    }

    /**
     * 행 내용 전체의 64-bit FNV-1a 해시.
     * Python hash(tuple(...))와 동급의 충돌 저항(64-bit) — 수만 행 규모에서 충돌 확률 사실상 0.
     * (기존 List.hashCode() 기반 32-bit 지문은 ~2만 행 수집 시 회당 ~5% 확률로
     *  서로 다른 행이 중복 판정되어 조용히 누락되는 문제가 있었음)
     * key/value 경계는 0x1F(unit separator)로 보존 — Python 튜플 (k, v)의 경계 의미론과 동일.
     */
    private static long rawFingerprint(Map<String, Object> raw) {
        List<String> entries = new ArrayList<>(raw.size());
        for (var e : raw.entrySet()) entries.add(e.getKey() + '\u001F' + e.getValue());
        Collections.sort(entries);
        long h = 0xcbf29ce484222325L; // FNV-1a offset basis
        for (String s : entries) {
            for (int i = 0; i < s.length(); i++) {
                h ^= s.charAt(i);
                h *= 0x100000001b3L; // FNV-1a prime
            }
            h ^= 0x1E; // 엔트리 구분자 (record separator)
            h *= 0x100000001b3L;
        }
        return h;
    }

    private static String toApiQuarter(String q) {
        if (q == null) return null;
        if (q.length() == 6 && q.charAt(4) == 'Q') return q.substring(0, 4) + q.charAt(5);
        if (q.length() == 5 && q.chars().allMatch(Character::isDigit)) return q;
        throw new IllegalArgumentException("unknown quarter format: " + q);
    }
}
