# K/V Cache Ablation: Paired Sample-Level and Bootstrap Analysis

- Generated: 2026-08-07T01:15:44.396320+08:00
- Git commit: `6fd713fe03cb53d7c998f9564db78b2b8ead80f2`
- Bootstrap iterations: 10000, base seed: 42

本分析以既有的五組完整 prediction JSONL 為輸入，重算 sample-level 分數並驗證與 `result.json` 一致，再以固定 15-task、task 內 paired bootstrap 為 primary 方法估計不確定性。

**配對方式**：因 prediction 不包含原始 sample ID，本分析以相同 task 內的 row order 配對，並以 answers、all_classes、length 的逐列一致性作為配對驗證（結果：PASS）。

## A. 直接觀察（五組方法平均）

| method | official average (result.json, rounded task scores) |
|---|---|
| FP16 | 38.5026667 |
| K2/V16 | 38.5893333 |
| K16/V2 | 38.2540000 |
| K2/V2 | 38.0186667 |
| K4/V4 | 38.6400000 |

- K2/V16 的平均（38.5893）接近 FP16（38.5027）。
- K16/V2 的平均（38.2540）比 FP16 低約 0.2487。
- K2/V2 的平均（38.0187）比 FP16 低約 0.4840。

## B. Overall paired effects — Primary (task-stratified) bootstrap

| effect | observed | 95% CI | sign probability | CI vs 0 |
|---|---|---|---|---|
| Key-only effect (K2/V16 - FP16) | +0.0842 | [-0.4321, +0.6086] | 0.7480 | CI 跨越 0（不穩定） |
| Value-only effect (K16/V2 - FP16) | -0.2499 | [-0.5101, +0.0216] | 0.0732 | CI 跨越 0（不穩定） |
| Joint effect (K2/V2 - FP16) | -0.4833 | [-0.9608, +0.0021] | 0.0512 | CI 跨越 0（不穩定） |
| KIVI-4 effect (K4/V4 - FP16) | +0.1368 | [-0.1135, +0.3919] | 0.2814 | CI 跨越 0（不穩定） |
| Key-only minus Value-only (K2/V16 - K16/V2) | +0.3340 | [-0.2038, +0.8563] | 0.2164 | CI 跨越 0（不穩定） |
| Interaction (K2/V2 - K2/V16 - K16/V2 + FP16) | -0.3176 | [-0.7354, +0.0864] | 0.1262 | CI 跨越 0（不穩定） |

Secondary (task-level resampling) bootstrap, for comparison only:

| effect | observed | secondary 95% CI |
|---|---|---|
| Key-only effect (K2/V16 - FP16) | +0.0842 | [-0.6769, +1.1152] |
| Value-only effect (K16/V2 - FP16) | -0.2499 | [-1.0596, +0.2907] |
| Joint effect (K2/V2 - FP16) | -0.4833 | [-0.9229, -0.0151] |
| KIVI-4 effect (K4/V4 - FP16) | +0.1368 | [-0.0917, +0.4481] |
| Key-only minus Value-only (K2/V16 - K16/V2) | +0.3340 | [-0.6913, +1.5828] |
| Interaction (K2/V2 - K2/V16 - K16/V2 + FP16) | -0.3176 | [-1.2105, +0.4402] |

## C. Leave-one-task-out

| effect | full 15-task mean | min (task removed) | max (task removed) | sign changed by removing any one task |
|---|---|---|---|---|
| Key-only effect (K2/V16 - FP16) | +0.0842 | -0.3384 (passage_retrieval_en) | +0.2191 (2wikimqa) | lcc, passage_retrieval_en |
| Value-only effect (K16/V2 - FP16) | -0.2499 | -0.3391 (passage_retrieval_en) | +0.0954 (lcc) | lcc |
| Joint effect (K2/V2 - FP16) | -0.4833 | -0.6428 (passage_retrieval_en) | -0.3868 (multifieldqa_en) | none |
| KIVI-4 effect (K4/V4 - FP16) | +0.1368 | +0.0037 (passage_retrieval_en) | +0.1837 (lcc) | none |
| Key-only minus Value-only (K2/V16 - K16/V2) | +0.3340 | -0.0985 (lcc) | +0.5070 (multifieldqa_en) | lcc |
| Interaction (K2/V2 - K2/V16 - K16/V2 + FP16) | -0.3176 | -0.5597 (lcc) | +0.0347 (passage_retrieval_en) | passage_retrieval_en |

- 移除 `lcc` 後 Value-only effect：+0.0954（轉為非負）。
- 移除 `passage_retrieval_en` 後各 effect：Key-only effect (K2/V16 - FP16)=-0.3384, Value-only effect (K16/V2 - FP16)=-0.3391, Joint effect (K2/V2 - FP16)=-0.6428, KIVI-4 effect (K4/V4 - FP16)=+0.0037, Key-only minus Value-only (K2/V16 - K16/V2)=+0.0008, Interaction (K2/V2 - K2/V16 - K16/V2 + FP16)=+0.0347
- Key-only 減 Value-only 的方向在 leave-one-task-out 下在移除某些 task 後變號，方向不穩定。
- Interaction 的符號會因移除單一 task 而改變，顯示可能由少數 task 主導。

## D. 敏感任務排序

- Value-only 最負面 5 個 task：lcc, qasper, repobench-p, multi_news, qmsum
- Key-only 最負面 5 個 task：2wikimqa, multifieldqa_en, repobench-p, triviaqa, musique
- Joint 最負面 5 個 task：multifieldqa_en, repobench-p, 2wikimqa, triviaqa, qasper
- Interaction 最負面 5 個 task：passage_retrieval_en, gov_report, multifieldqa_en, triviaqa, trec
- K2/V16 與 K16/V2 差異最大 5 個 task：lcc, passage_retrieval_en, multifieldqa_en, 2wikimqa, narrativeqa
- Value-only 95% CI 完全低於 0 的 task：lcc
- Key-only 95% CI 完全低於 0 的 task：2wikimqa
- Interaction 95% CI 完全低於 0 的 task：gov_report, passage_retrieval_en

特別關注任務（觀察值，不預設顯著）：

| task | key_only | value_only | joint | kivi4 | key_minus_value | interaction |
|---|---|---|---|---|---|---|
| lcc | +1.3060 | -5.0840 | -0.7060 | -0.5200 | +6.3900 | +3.0720 |
| repobench-p | -1.3140 | -0.7960 | -1.6240 | -0.2740 | -0.5180 | +0.4860 |
| passage_retrieval_en | +6.0000 | +1.0000 | +1.7500 | +2.0000 | +5.0000 | -5.2500 |
| 2wikimqa | -1.8039 | -0.0184 | -1.5179 | +0.1507 | -1.7855 | +0.3043 |
| multifieldqa_en | -1.5581 | +0.5293 | -1.8342 | +0.2489 | -2.0874 | -0.8054 |

## E. Bootstrap 支持程度

- Key-only effect 的 primary CI 跨越 0，與觀察到「幾乎無損」的方向一致。
- Value-only effect 的 primary CI 跨越 0。
- Interaction 的 primary CI 跨越 0。
- Value-only 比 Key-only 敏感的方向在 leave-one-task-out 下不穩定，會因移除單一 task 而改變。

## F. 不可宣稱的內容（限制）

1. 每組方法只有一批 deterministic greedy predictions（無多 seed 重複）。
2. Bootstrap 估計的是 benchmark samples/tasks 的不確定性，不是模型重新執行的 run-to-run variance，也不是跨硬體變異的估計。
3. Prediction 無 sample ID，配對依賴已驗證的 row order 一致性。
4. LongBench 15-task 平均差距很小，不等於普遍模型品質差異。
5. 本分析未涵蓋其他模型、context length、硬體或 bit configuration。
6. 不能只依此證明 Value 量化在所有模型上都比 Key 量化敏感。
7. Interaction 為 operational ablation interaction 的估計，不是因果機制證明。

## G. 結論

在本次 LongChat-7B / LongBench 設定下，結果初步支持：Value-only 量化（K16/V2）比 Key-only 量化（K2/V16）對整體平均分數的影響更大，且 joint（K2/V2）量化的降幅明顯大於兩個單側效果的簡單相加，初步支持存在負向 interaction。但整體平均層級的差距小於個別 task 的波動幅度（尤其 `lcc` 與 `passage_retrieval_en`），尚需跨模型與重複實驗驗證，才能將 task 層級的觀察（例如 code-completion 類任務對 Value 量化較敏感）視為穩定結論。
