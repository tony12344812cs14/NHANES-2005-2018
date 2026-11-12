# 外部資料：討論紀錄（2026-11-12）

目標：找 NHANES 以外的資料，拿來做模型的外部驗證。這份文件記錄考慮過的選項、每個選項的結論，以及 NHIS 的盤點結果。

## 結論

- **外部驗證用 NHIS**（美國國民健康訪問調查）。KNHANES 和 NHANES 最像，但註冊需要韓國手機號碼，目前拿不到。
- NHANES 其他年份（2021–2023）**不算外部資料**（老師的意見），不使用。
- 不從外部資料加特徵（理由見第 2 節）。
- NHIS 只有問卷題目。317 個特徵中約 74 個對得上，檢驗、血壓、飲食全部沒有。所以 NHIS 驗證的對象是**只用問卷的精簡模型**。
- 下一步（待確認）：先只用 NHANES，以這約 70 個特徵加 PHQ-8 標籤跑 LOCO，看精簡模型比全特徵模型低多少，再決定 NHIS 驗證值不值得做。

## 1. 外部資料的兩種用途

| 用途 | 意思 | 結論 |
|---|---|---|
| 外部驗證 | 在另一個資料集測模型，看能不能類推 | **採用** |
| 加特徵 | 把外部資料串到每個受訪者身上，提高預測力 | 不採用 |

## 2. 為什麼不從外部資料加特徵

- 公開版 NHANES 沒有州、郡、郵遞區號，所以空汙、區域貧困指數（ADI）、失業率這類地區資料串不上。要拿到地理碼必須申請 NCHS 的 RDC（受限資料中心），通常要幾個月。
- 公開檔能串的只剩年度或全國層級的資料。這類資料在同一個 cycle 裡每個人都一樣，等於是 cycle 標記，正是 step 0 要找出來處理的那種差異。在 LOCO 下，被留出的 cycle 還會出現訓練時沒看過的值。
- NHANES 串死亡登記（NDI）的檔案記錄的是受訪之後的結果，不能當特徵。

## 3. 參考的 5 篇論文都沒有使用外部資料

| 論文 | 資料 | 驗證方式 |
|---|---|---|
| 1. Fairness and bias correction（Dang et al. 2024） | NHANES、UK Biobank、LONGSCAN、FUUS | 四個資料集各自做 k-fold CV，沒有用 A 訓練、B 測試 |
| 2. Passively-collected wearable data | NHANES 的加速規資料（屬於 NHANES 本身的元件） | 80/20 切分加 10-fold CV |
| 3. PFAS | NHANES 血清 PFAS | 80/20 切分，文末寫 "requires further external validation" |
| 4. MDD among adults living alone | NHANES | 10-fold CV，文末寫還需要進一步驗證 |
| 5. Physically inactive adults | NHANES 2005–2020 | 摘要寫 "three external cohorts"，內文其實是 "Internal validation sets"，也就是 NHANES 2005–06、2015–16、2019–20，和訓練資料重疊 |

這幾篇的「驗證」最多只到換 NHANES 的 cycle。我們的 LOCO 已經比它們嚴格，真正的外部驗證會是新的貢獻。

## 4. 考慮過的資料集

| 資料集 | 憂鬱量表 | 和我們特徵的重疊 | 取得方式 | 結論 |
|---|---|---|---|---|
| NHANES 2021–2023 | PHQ-9 | 高 | 公開 | 不算外部資料，**不用** |
| KNHANES（韓國） | PHQ-9，2014 年起隔年調查（2014、2016、2018、2020 確定有） | 高，仿 NHANES 設計，有檢驗、血壓、24 小時飲食 | KDCA 網站註冊，**需要韓國手機號碼** | 目前拿不到；如果能找人代下載，仍是首選 |
| **NHIS（美國）** | PHQ-8，2019、2022 年 | 低，只有問卷 | 公開直接下載 | **採用** |
| BRFSS（美國） | PHQ-8，2006、2008、2010 年，只有部分州 | 比 NHIS 更少 | 公開 | 備選 |
| CHARLS（中國） | CES-D-10（不同量表） | 有血檢、血壓，但只收 45 歲以上 | email 註冊 | 標籤定義不同，不優先 |
| UK Biobank | PHQ-9 | 中到高 | 要申請、付費，要等幾個月 | 時程不適合 |

已有文獻做過 NHANES → KNHANES 的外推：logistic regression 的 AUC 0.77，深度學習 0.74（[PMC9337649](https://pmc.ncbi.nlm.nih.gov/articles/PMC9337649)）。如果之後拿到 KNHANES，這可以當比較基準。

### 4.1 NHANES 2021–2023 的盤點（已放棄，只留紀錄）

- 族群：≥20 歲未懷孕 7,768 人，PHQ-9 九題全答 5,175 人，PHQ-9 ≥ 10 有 657 人（12.7%）。
- 317 個特徵中有 217 個找得到來源變數。
- 這一期沒有 CDQ、DUQ、PFQ，血壓只有示波式（`BPXO`），WHQ 只剩 5 個變數，處方藥檔沒有藥名。

## 5. NHIS 盤點

資料：2019 和 2022 的 Sample Adult 公開檔與 codebook，從 CDC FTP 下載（`Datasets/NHIS/<年>/adult<yy>csv.zip`）。

### 5.1 樣本

族群條件和我們一樣：≥20 歲、未懷孕、PHQ-8 八題全答。

| | NHIS 2019 | NHIS 2022 | NHANES 2005–18 |
|---|---|---|---|
| 人數 | 30,334 | 26,290 | 33,498 |
| 憂鬱 | PHQ-8 ≥ 10：2,113（7.0%） | PHQ-8 ≥ 10：1,865（7.1%，加權 7.3%） | PHQ-9 ≥ 10：8.9% |

- NHIS 只有 PHQ-8，少了第 9 題（自傷念頭）。NHANES 這邊改用 PHQ-8 ≥ 10 當訓練標籤，兩邊的標籤就一致。PHQ-8 ≥ 10 是文獻常用的切點。
- 計分方式：NHIS 的選項代碼是 1–4，每題要減 1，才和 NHANES 的 0–3 對齊。

### 5.2 特徵對照摘要

| 類別 | 能用的 | 不能用的 |
|---|---|---|
| 人口學、社經 | 年齡、性別、種族、教育、婚姻、公民身分、出生地、在美年數、家庭人數、貧窮比、收入、兵役 | 訪談語言、戶長資訊 |
| 健康與就醫 | 自評健康、固定就醫地點、住院、保險有無與種類、過去一年沒保險 | 就醫次數 |
| 慢性病 | 高血壓與用藥、高膽固醇與用藥、糖尿病（含前期、胰島素、口服藥）、氣喘、關節炎、冠心病、心絞痛、心肌梗塞、中風、COPD、癌症 | 心衰竭、肝病、甲狀腺、家族史、貧血、腎臟、漏尿 |
| 生活型態 | 抽菸狀態與支數；只有 2022：飲酒頻率、每次杯數、豪飲、睡眠時數 | 尼古丁替代品、家中有人抽菸、毒品 |
| 功能與工作 | 工作受限、行動輔具、記憶困難、上週工時、沒工作的原因、住宅自有或租屋 | |
| 體位 | 身高、體重、BMI（NHIS 自填，NHANES 實測） | 腰圍、臂圍等 |
| 檢驗、血壓、飲食 | | 全部 147 個 |

- 能對上的 74 個特徵中，定義相同的 37 個，要轉換的 37 個。
- 兩個年份各對得上 70 個，但組成不同：2022 有酒精和睡眠；2019 有膽固醇檢查時間、子宮切除、血糖檢查。**以 2022 為主。**
- NHIS 的「曾被診斷憂鬱 / 焦慮」（`DEPEV_A`、`ANXEV_A`）、抗憂鬱藥（`DEPMED_A`）、心理健康用藥與治療（`MHRX_A`、`MHTHRPY_A`）、GAD-7、憂鬱 / 焦慮頻率（`DEPFREQ_A`、`ANXFREQ_A`）等，要照 pipeline step 2 的規則排除，避免洩漏答案。

### 5.3 逐項對照表

只看變數名稱和 codebook 的題目文字，代碼還沒有逐一核對。

| 我們的特徵 | NHIS 2019 | NHIS 2022 | 對應 | 說明 |
|---|---|---|---|---|
| `RIDAGEYR` | `AGEP_A` | `AGEP_A` | 相同 | NHIS 上限 85，NHANES 80 |
| `RIAGENDR` | `SEX_A` | `SEX_A` | 相同 |  |
| `RIDRETH1` | `HISPALLP_A` | `HISPALLP_A` | 要轉換 | 合併成 NHANES 的 5 組 |
| `DMDEDUC2` | `EDUC_A` | `EDUCP_A` | 要轉換 | 合併成 5 級 |
| `DMDMARTL` | `MARSTAT_A` | `MARSTAT_A` | 要轉換 | 由婚姻現況加法定婚姻狀態組成 NHANES 的 6 類 |
| `DMDCITZN` | `CITZNSTP_A` | `CITZNSTP_A` | 相同 |  |
| `born_us` | `NATUSBORN_A` | `NATUSBORN_A` | 要轉換 | NHIS 把美國屬地也算美國出生 |
| `years_in_us` | `YRSINUS_A` | `YRSINUS_A` | 要轉換 | 分組切點不同 |
| `DMDFMSIZ` | `PCNTFAM_A` | `PCNTADLT_A+PCNTKIDS_A` | 要轉換 | NHIS 上限 6+（NHANES 7+） |
| `DMDHHSIZ` | `PCNTTC` | `PCNT18UPTC+PCNTLT18TC` | 要轉換 | 上限不同 |
| `INDFMPIR` | `POVRATTC_A` | `POVRATTC_A` | 要轉換 | 比照 NHANES 截在 5 |
| `fam_income_cat` | `FAMINCTC_A` | `FAMINCTC_A` | 要轉換 | 金額換成 NHANES 的級距 |
| `military_service` | `AFVET_A` | `AFVET_A` | 相同 | 現役（同 NHANES 2011 起的問法） |
| `RIDEXMON` | `INTV_QRT` | `INTV_QRT` | 要轉換 | 季度只能粗略對到 11–4 月 / 5–10 月 |
| `hh_spouse_edu3` | `SPOUSEDUC_A` | `SPOUSEDUCP_A` | 要轉換 | 是受訪者的配偶，不是戶長的配偶 |
| `HSD010` | `PHSTAT_A` | `PHSTAT_A` | 相同 |  |
| `HUQ010` | `PHSTAT_A` | `PHSTAT_A` | 相同 | 和 HSD010 同一題 |
| `HUQ030` | `USUALPL_A` | `USUALPL_A` | 相同 |  |
| `usual_care_place` | `USPLKIND_A` | `USPLKIND_A` | 要轉換 |  |
| `HUQ071` | `HOSPONGT_A` | `HOSPONGT_A` | 相同 |  |
| `HIQ011` | `HICOV_A` | `HICOV_A` | 相同 |  |
| `HIQ031A` | `HIKIND01_A` | `HIKIND01_A` | 相同 |  |
| `HIQ031B` | `HIKIND02_A` | `HIKIND02_A` | 相同 |  |
| `HIQ031C` | `HIKIND03_A` | `HIKIND03_A` | 相同 |  |
| `HIQ031D` | `HIKIND04_A` | `HIKIND04_A` | 相同 |  |
| `HIQ031F` | `HIKIND06_A` | `HIKIND06_A` | 相同 |  |
| `HIQ031H` | `HIKIND08_A` | `HIKIND08_A` | 相同 |  |
| `HIQ031I` | `HIKIND09_A` | `HIKIND09_A` | 相同 |  |
| `insurance_type` | `COVER_A/COVER65_A` | `COVER_A/COVER65_A` | 要轉換 |  |
| `HIQ210` | `HINOTYR_A` | `HINOTYR_A` | 要轉換 |  |
| `HIQ270` | `PRRXCOV1_A` | `PRRXCOV1_A` | 要轉換 | 只有私人保險有問 |
| `htn_dx` | `HYPEV_A` | `HYPEV_A` | 相同 |  |
| `htn_med` | `HYPMED_A` | `HYPMED_A` | 相同 |  |
| `chol_dx` | `CHLEV_A` | `CHLEV_A` | 相同 |  |
| `chol_med` | `CHLMED_A` | `CHLMED_A` | 相同 |  |
| `chol_last_checked` | `CHOLLAST_A` | — | 要轉換 | 只有 2019 |
| `chol_checked` | `CHOLLAST_A` | — | 要轉換 | 只有 2019 |
| `diabetes_status` | `DIBEV_A+PREDIB_A` | `DIBEV_A+PREDIB_A` | 要轉換 | 沒有「邊緣」選項，用前期糖尿病代替 |
| `DIQ160` | `PREDIB_A` | `PREDIB_A` | 相同 |  |
| `DIQ180` | `DIABLAST_A` | — | 要轉換 | 只有 2019 |
| `insulin` | `DIBINS_A` | `DIBINS_A` | 要轉換 | NHIS 只問糖尿病患者 |
| `diabetes_pills` | `DIBPILL_A` | `DIBPILL_A` | 相同 |  |
| `MCQ010` | `ASEV_A` | `ASEV_A` | 相同 |  |
| `asthma_current` | `ASTILL_A` | `ASTILL_A` | 相同 |  |
| `MCQ160A` | `ARTHEV_A` | `ARTHEV_A` | 相同 |  |
| `MCQ160C` | `CHDEV_A` | `CHDEV_A` | 相同 |  |
| `MCQ160D` | `ANGEV_A` | `ANGEV_A` | 相同 |  |
| `MCQ160E` | `MIEV_A` | `MIEV_A` | 相同 |  |
| `MCQ160F` | `STREV_A` | `STREV_A` | 相同 |  |
| `MCQ160G` | `COPDEV_A` | `COPDEV_A` | 要轉換 | NHIS 把 COPD、肺氣腫、慢性支氣管炎合成一題 |
| `MCQ160K` | `COPDEV_A` | `COPDEV_A` | 要轉換 | 同上，合併的那一題 |
| `MCQ220` | `CANEV_A` | `CANEV_A` | 相同 |  |
| `PFQ049` | `SOCWRKLIM_A` | `SOCWRKLIM_A` | 要轉換 |  |
| `PFQ054` | `EQUIP_A` | `EQUIP_A` | 要轉換 |  |
| `PFQ057` | `COGMEMDFF_A` | `COGMEMDFF_A` | 要轉換 |  |
| `pf_difficulty_count` | `Washington Group 題組` | `Washington Group 題組` | 要轉換 | 活動項目不同（Washington Group 題組） |
| `RHD280` | `HYSTEV_A` | — | 相同 | 只有 2019 |
| `RHQ131` | `LIVEBIRTH_A` | `LIVEBIRTH_A` | 要轉換 | 問活產，不是曾經懷孕 |
| `smoking_status` | `SMKCIGST_A` | `SMKCIGST_A` | 相同 |  |
| `cigs_per_day` | `CIGNOW_A` | `CIGNOW_A` | 要轉換 |  |
| `alcohol_freq` | — | `DRK12MN_A/DRK12MTP_A` | 要轉換 | 只有 2022 |
| `drinks_per_day` | — | `DRKAVG12M_A` | 相同 | 只有 2022 |
| `binge_freq` | — | `DRKBNG30D_A` | 要轉換 | 只有 2022；期間是 30 天，不是 12 個月 |
| `sleep_hours` | — | `SLPHOURS_A` | 要轉換 | 只有 2022；問一天 24 小時，不是平日晚上 |
| `hours_worked` | `EMPWKHRS2_A` | `EMPWKHRS3_A` | 相同 |  |
| `OCD150` | `EMPWRKLSWK_A` | `EMPLASTWK_A` | 要轉換 |  |
| `not_working_reason` | `EMPRSNOWK_A` | `EMPWHYNOT_A` | 要轉換 |  |
| `HOQ065` | `HOUTENURE_A` | `HOUTENURE_A` | 相同 |  |
| `WHD010` | `HEIGHTTC_A` | `HEIGHTTC_A` | 相同 | 兩邊都是自填 |
| `WHD020` | `WEIGHTLBTC_A` | `WEIGHTLBTC_A` | 相同 | 兩邊都是自填 |
| `BMXBMI` | `HEIGHTTC_A+WEIGHTLBTC_A` | `HEIGHTTC_A+WEIGHTLBTC_A` | 要轉換 | NHANES 實測，NHIS 自填 |
| `BMXHT` | `HEIGHTTC_A` | `HEIGHTTC_A` | 要轉換 | 實測對自填 |
| `BMXWT` | `WEIGHTLBTC_A` | `WEIGHTLBTC_A` | 要轉換 | 實測對自填 |
| `rx_count` | `RX12M_A` | `RX12M_A` | 要轉換 | 只有過去 12 個月有沒有用處方藥 |

## 6. 下一步（待確認）

1. 在 NHANES 上用上表的特徵（以 2022 為準），標籤改成 PHQ-8 ≥ 10，跑 LOCO，和現有的全特徵結果比較。
2. 如果精簡模型的表現還可以：逐項核對 NHIS 的代碼並做轉換，在 NHANES 2005–18 全部資料上訓練，拿到 NHIS 2022 測試。報告 AUROC，並另外報告校準。
3. KNHANES：如果找得到有韓國帳號的人代下載（先確認使用條款允許轉交），再補做一次盤點。
