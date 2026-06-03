package com.salesmap.ai.service;

import com.salesmap.ai.core.Settings;
import com.salesmap.ai.dto.IndustryCategory;
import com.salesmap.ai.model.SalesRecord;
import com.salesmap.ai.predictor.NotEnoughDataException;
import com.salesmap.ai.predictor.Predictor;
import com.salesmap.ai.predictor.PredictorFactory;
import com.salesmap.ai.repository.PredictionRepository;
import com.salesmap.ai.repository.SalesRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 셀(region × industry) 단위 선형회귀 예측 생성. Predictor는 Factory가 생성한 Strategy 객체. */
@Service
public class PredictionGenerateService {

    private static final Logger log = LoggerFactory.getLogger(PredictionGenerateService.class);
    private static final List<IndustryCategory> INDUSTRIES =
        List.of(IndustryCategory.food, IndustryCategory.service, IndustryCategory.retail);

    private final SalesRepository salesRepo;
    private final PredictionRepository predRepo;
    private final Settings settings;

    public PredictionGenerateService(SalesRepository salesRepo, PredictionRepository predRepo, Settings settings) {
        this.salesRepo = salesRepo;
        this.predRepo = predRepo;
        this.settings = settings;
    }

    public PredictionRepository predRepo() { return predRepo; }

    public CellPrediction predictCell(int regionId, String industry, String targetQuarter, int lookback) {
        List<SalesRecord> records = salesRepo.findQuarters(regionId, industry, lookback);
        if (records.size() < 2) {
            throw new NotEnoughDataException("need >= 2 quarters, got " + records.size());
        }

        String[] quarters = records.stream().map(SalesRecord::getQuarter).toArray(String[]::new);
        long[] sales = records.stream().mapToLong(SalesRecord::getTotalSales).toArray();
        int rawN = records.size();

        FillResult ff = forwardFillQuarters(quarters, sales);
        quarters = ff.quarters;
        sales = ff.sales;
        ClipResult cr = clipOutliersIqr(sales, 1.5);
        sales = cr.values;
        if (ff.filled > 0 || cr.clipped > 0) {
            log.info("preprocess_applied region_id={} industry={} raw_samples={} final_samples={} filled_quarters={} clipped_values={}",
                regionId, industry, rawN, sales.length, ff.filled, cr.clipped);
        }

        int[] xs = Arrays.stream(quarters).mapToInt(Quarter::toIndex).toArray();
        String latestQ = quarters[quarters.length - 1];
        String tq = (targetQuarter == null || targetQuarter.isBlank()) ? Quarter.nextQuarter(latestQ) : targetQuarter;

        Predictor predictor = PredictorFactory.create(settings.getPredictor());
        predictor.train(xs, sales);
        double predicted = predictor.predict(Quarter.toIndex(tq));
        var params = predictor.params();

        // Python round()와 동일한 banker's rounding (half-to-even)
        long predictedSales = Math.max((long) Math.rint(predicted), 0L);
        return new CellPrediction(
            regionId,
            industry,
            tq,
            predictedSales,
            records.get(records.size() - 1).getTotalSales(),
            params.get("slope"),
            params.get("intercept"),
            records.size(),
            records.get(0).getQuarter(),
            latestQ
        );
    }

    public BatchResult runBatch(List<Integer> regionIds, List<String> industries, String targetQuarter, int lookback) {
        List<Integer> regions = (regionIds == null || regionIds.isEmpty()) ? salesRepo.listRegionIds() : regionIds;
        List<String> inds = (industries == null || industries.isEmpty())
            ? INDUSTRIES.stream().map(Enum::name).toList()
            : industries;

        int processed = 0, succeeded = 0, failed = 0;
        List<CellFailure> failures = new ArrayList<>();
        String resolvedTarget = targetQuarter == null ? "" : targetQuarter;

        for (int regionId : regions) {
            for (String industry : inds) {
                processed++;
                CellPrediction cell;
                try {
                    cell = predictCell(regionId, industry, targetQuarter, lookback);
                } catch (NotEnoughDataException e) {
                    failed++;
                    failures.add(new CellFailure(regionId, industry, "NO_SALES_DATA:" + e.getMessage()));
                    continue;
                } catch (RuntimeException e) {
                    failed++;
                    failures.add(new CellFailure(regionId, industry, "ERROR:" + e.getMessage()));
                    continue;
                }

                predRepo.save(
                    cell.regionId(),
                    cell.industry(),
                    cell.targetQuarter(),
                    cell.predictedSales(),
                    cell.previousSales(),
                    cell.slope(),
                    cell.intercept(),
                    cell.samplesUsed()
                );
                resolvedTarget = cell.targetQuarter();
                succeeded++;
            }
        }

        return new BatchResult(
            (!resolvedTarget.isEmpty()) ? resolvedTarget : (targetQuarter != null ? targetQuarter : "unknown"),
            processed, succeeded, failed, failures
        );
    }

    record FillResult(String[] quarters, long[] sales, int filled) {}
    record ClipResult(long[] values, int clipped) {}

    /** records 사이 빠진 분기를 직전 값으로 채움. */
    static FillResult forwardFillQuarters(String[] quarters, long[] sales) {
        if (quarters.length < 2) return new FillResult(quarters, sales, 0);
        int startIdx = Quarter.toIndex(quarters[0]);
        int endIdx = Quarter.toIndex(quarters[quarters.length - 1]);
        java.util.Map<Integer, Long> known = new java.util.HashMap<>();
        for (int i = 0; i < quarters.length; i++) known.put(Quarter.toIndex(quarters[i]), sales[i]);

        int size = endIdx - startIdx + 1;
        String[] outQ = new String[size];
        long[] outV = new long[size];
        int filled = 0;
        long lastVal = sales[0];
        for (int i = 0; i < size; i++) {
            int idx = startIdx + i;
            outQ[i] = Quarter.fromIndex(idx);
            if (known.containsKey(idx)) lastVal = known.get(idx);
            else filled++;
            outV[i] = lastVal;
        }
        return new FillResult(outQ, outV, filled);
    }

    /** 튜키 IQR clip. n<8 또는 IQR=0이면 패스. Python statistics.quantiles(n=4) 와 동일 보간. */
    static ClipResult clipOutliersIqr(long[] values, double k) {
        int n = values.length;
        if (n < 8) return new ClipResult(values, 0);
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        double q1 = quantile(sorted, 1);
        double q3 = quantile(sorted, 3);
        double iqr = q3 - q1;
        if (iqr == 0.0) return new ClipResult(values, 0);
        double lo = q1 - k * iqr;
        double hi = q3 + k * iqr;
        int clipped = 0;
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            long v = values[i];
            if (v < lo || v > hi) clipped++;
            double cv = Math.min(Math.max((double) v, lo), hi);
            // Python int(round(...))와 동일한 banker's rounding
            out[i] = (long) Math.rint(cv);
        }
        return new ClipResult(out, clipped);
    }

    /** Python statistics.quantiles(data, n=4)[k-1] 동등물: 위치 m = k*(n+1)/4. */
    private static double quantile(long[] sorted, int k) {
        int n = sorted.length;
        double m = (double) k * (n + 1) / 4.0;
        int j = (int) Math.floor(m);
        double frac = m - j;
        if (j < 1) return sorted[0];
        if (j >= n) return sorted[n - 1];
        return sorted[j - 1] + frac * (sorted[j] - sorted[j - 1]);
    }
}
