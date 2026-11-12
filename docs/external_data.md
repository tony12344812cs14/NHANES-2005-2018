# 外部資料：討論紀錄（2026-11-12）

目標：找 NHANES 以外的資料，拿來做模型的外部驗證。這份文件記錄考慮過的選項、每個選項的結論，以及 NHIS 和 KNHANES 的盤點結果。

## 結論

- 有兩個外部資料集可用：
  - **KNHANES**（韓國）：已取得 2008–2017 年的檔案。有 PHQ-9 的是 2014 和 2016 兩年，共 10,518 人。317 個特徵中有 125 個對得上，包含檢驗、血壓、體位、飲食。
  - **NHIS**（美國）：只有問卷題目。317 個特徵中約 74 個對得上，檢驗、血壓、飲食全部沒有。
- **建議以 KNHANES 為主**（待確認）。它的特徵重疊比 NHIS 多，憂鬱量表也一樣是 PHQ-9。NHIS 當第二個外部資料集。
- NHANES 其他年份（2021–2023）**不算外部資料**（老師的意見），不使用。
- 不從外部資料加特徵（理由見第 2 節）。
- 只用 KNHANES 對得上的特徵，NHANES 內部 LOCO 的 AUROC 從 0.874 降到 0.838（step 9）。下一步是核對代碼、轉換定義，做真正的外部測試（第 7 節）。

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
| **KNHANES（韓國）** | PHQ-9，2014 年起隔年調查（我們手上的檔案中只有 2014、2016 有） | 高，仿 NHANES 設計，有檢驗、血壓、24 小時飲食 | KDCA 網站註冊，需要韓國手機號碼；**已取得 2008–2011、2013–2017** | **建議為主**（第 6 節） |
| **NHIS（美國）** | PHQ-8，2019、2022 年 | 低，只有問卷 | 公開直接下載 | **採用，當第二個資料集**（第 5 節） |
| BRFSS（美國） | PHQ-8，2006、2008、2010 年，只有部分州 | 比 NHIS 更少 | 公開 | 備選 |
| CHARLS（中國） | CES-D-10（不同量表） | 有血檢、血壓，但只收 45 歲以上 | email 註冊 | 標籤定義不同，不優先 |
| UK Biobank | PHQ-9 | 中到高 | 要申請、付費，要等幾個月 | 時程不適合 |

已有文獻做過 NHANES → KNHANES 的外推：logistic regression 的 AUC 0.77，深度學習 0.74（[PMC9337649](https://pmc.ncbi.nlm.nih.gov/articles/PMC9337649)），可以當 KNHANES 驗證的比較基準。

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

### 5.3 逐項對照表（NHIS）

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

## 6. KNHANES 盤點

資料：從 KDCA 網站下載的基本資料檔（`HN<yy>_ALL`，SAS 格式），年份是 2008–2011、2013–2017。變數標籤直接從 SAS 檔讀出（編碼是 cp949）。對照表中的變數名稱都已確認存在於檔案裡；代碼還要對照 KNHANES 的使用指南（이용지침서）逐一核對。

### 6.1 哪些年份有 PHQ-9

| 年份 | 憂鬱相關變數 |
|---|---|
| 2008–2011、2013、2015、2017 | 沒有 PHQ-9，只有單題（例如 `BP5`「連續兩週以上憂鬱感」）和憂鬱症診斷（`DF2_*`） |
| **2014、2016** | **PHQ-9 九題**（`BP_PHQ_1`–`BP_PHQ_9`，0–3 分），加上總分 `mh_PHQ_S`（和九題加總完全一致） |

2018、2020、2022 的檔案還沒下載。KNHANES 是隔年收 PHQ-9，這幾年應該也有，需要再確認。如果有，樣本可以增加一倍以上。

### 6.2 樣本

族群條件和我們一樣：≥20 歲、目前未懷孕（`HE_prg`）、PHQ-9 九題全答。`LW_pr` 是「曾經懷孕」，不能拿來排除目前懷孕的人。

| | KNHANES 2014 | KNHANES 2016 | 合計 |
|---|---|---|---|
| ≥20 歲 | 5,897 | 6,315 | |
| 目前未懷孕 | 5,865 | 6,276 | |
| PHQ-9 九題全答 | 4,852 | 5,666 | 10,518 |
| PHQ-9 ≥ 10 | 342（7.0%，加權 6.8%） | 351（6.2%，加權 5.6%） | 693 |

- 加權用的是 `wt_itvex`（問卷加健檢的權重）。
- 盛行率比 NHANES 的 8.9% 低。這可能是族群差異，也可能是文化上作答方式不同，報告時要說明。
- 盛行率不同不影響 AUROC，但會影響校準。

### 6.3 特徵對照摘要

| 類別 | 能用的 | 不能用的 |
|---|---|---|
| 人口學、社經 | 年齡、性別、教育、婚姻、家戶人數、收入 | 種族（全是韓國人）、公民身分、出生地、貧窮比、兵役、訪談語言 |
| 健康與就醫 | 自評健康、住院（只有 2016）、私人保險 | 固定就醫地點、美國式的保險種類 |
| 慢性病 | 高血壓與用藥、血脂異常與用藥、糖尿病（含胰島素、口服藥）、氣喘、關節炎（含類型）、心肌梗塞、心絞痛、中風、COPD、肝病、甲狀腺、癌症、腎衰竭、家族史（心臟病、糖尿病） | 心衰竭、貧血治療、漏尿、胸痛 |
| 生活型態 | 抽菸狀態與支數、家中二手菸、飲酒頻率、每次杯數、豪飲、睡眠時數（只有 2016） | 毒品、尼古丁產品 |
| 功能與工作 | 活動受限、工作狀態、工時、沒工作的原因、住宅自有 | 身體功能的 20 項活動 |
| 體重與生育 | 自覺體型、減重嘗試與方法、一年內體重變化；初經年齡、懷孕經驗與次數、口服避孕藥 | 各年齡的自報體重 |
| 體位、血壓 | 身高、體重、BMI、腰圍、三次收縮壓 / 舒張壓、脈搏 | 臂圍、腿長 |
| 檢驗 | AST、ALT、BUN、肌酸酐、總膽固醇、HDL、三酸甘油酯、血糖、HbA1c、尿酸（只有 2016）、尿肌酸酐、血球計數（WBC、RBC、Hb、Hct、MCV、MCH、MCHC、血小板）、cotinine（尿液） | 白蛋白、ALP、GGT、電解質、鐵、膽紅素、LDH、白血球分類 |
| 飲食 | 熱量、水分、蛋白質、脂肪（總、飽和、單元、多元不飽和）、碳水化合物、鈣、磷、鈉、鉀、鐵、β-胡蘿蔔素、視黃醇、B1、B2、菸鹼酸、維生素 C；只有 2016：膽固醇、纖維、糖、鎂、鋅、維生素 A（RAE）、維生素 E、葉酸 | 個別脂肪酸、咖啡因、類胡蘿蔔素細項、B6、B12、維生素 K、硒、銅 |

- 能對上的 125 個特徵中，定義相同的 50 個，要轉換的 75 個；兩年都有的 114 個，只有 2016 有的 11 個。
- 要特別注意的差異：
  - KNHANES 的血糖、三酸甘油酯是空腹檢體，NHANES 生化盤是非空腹。
  - 飲食在 KNHANES 只有 1 天回憶（NHANES 取 2 天平均），而且用韓國的食物成分表。
  - cotinine 在 KNHANES 是尿液，NHANES 是血清。
  - 血壓的量測儀器與流程要再核對。
- 要排除的變數（和 pipeline step 2 一樣，避免洩漏答案）：
  - 憂鬱症診斷與治療（`DF2_*`）、連續兩週憂鬱感（`BP5`）、自殺意念 / 計畫 / 企圖（`BP6_*`）、心理諮商（`BP7`）。
  - 壓力感知（`BP1`、`mh_stress`）。
  - EQ-5D 的焦慮 / 憂鬱題（`LQ_5EQL`）和 EQ-5D 總分（`EQ5D`）。
  - 因憂鬱 / 焦慮而活動受限（`LQ4_15`）。

### 6.4 逐項對照表

| 我們的特徵 | KNHANES | 年份 | 對應 | 說明 |
|---|---|---|---|---|
| `RIDAGEYR` | `age` | 2014、2016 | 相同 | KNHANES 上限 80 |
| `RIAGENDR` | `sex` | 2014、2016 | 相同 |  |
| `DMDEDUC2` | `edu / educ` | 2014、2016 | 要轉換 | 學制不同，合併成 4–5 級 |
| `DMDMARTL` | `marri_1 + marri_2` | 2014、2016 | 要轉換 |  |
| `DMDHHSIZ` | `cfam` | 2014、2016 | 相同 |  |
| `DMDFMSIZ` | `cfam` | 2014、2016 | 要轉換 | KNHANES 只有家戶人數 |
| `fam_income_cat` | `ainc` | 2014、2016 | 要轉換 | 韓元月收入，要換成相對級距 |
| `hh_income_cat` | `ainc` | 2014、2016 | 要轉換 | 同上 |
| `HSD010` | `D_1_1` | 2014、2016 | 相同 |  |
| `HUQ010` | `D_1_1` | 2014、2016 | 相同 | 和 HSD010 同一題 |
| `HUQ071` | `MH1_yr` | 只有 2016 | 相同 | 只有 2016 |
| `HIQ031A` | `npins` | 2014、2016 | 要轉換 | 韓國是補充型私人保險，意義不同 |
| `htn_dx` | `DI1_dg` | 2014、2016 | 相同 |  |
| `htn_med` | `DI1_2` | 2014、2016 | 要轉換 |  |
| `chol_dx` | `DI2_dg` | 2014、2016 | 要轉換 | 問的是「血脂異常」 |
| `chol_med` | `DI2_2` | 2014、2016 | 相同 |  |
| `diabetes_status` | `DE1_dg` | 2014、2016 | 要轉換 | 沒有「邊緣」選項 |
| `insulin` | `DE1_31` | 2014、2016 | 要轉換 | 只問糖尿病患者 |
| `diabetes_pills` | `DE1_32` | 2014、2016 | 相同 |  |
| `MCQ010` | `DJ4_dg` | 2014、2016 | 相同 |  |
| `asthma_current` | `DJ4_pr` | 2014、2016 | 相同 |  |
| `MCQ160A` | `DM1_dg` | 2014、2016 | 相同 |  |
| `arthritis_type` | `DM2_dg + DM3_dg` | 2014、2016 | 要轉換 | 骨關節炎 / 類風濕 |
| `MCQ160C` | `DI4_dg` | 2014、2016 | 要轉換 | 心肌梗塞或心絞痛合併一題 |
| `MCQ160D` | `DI6_dg` | 2014、2016 | 相同 |  |
| `MCQ160E` | `DI5_dg` | 2014、2016 | 相同 |  |
| `MCQ160F` | `DI3_dg` | 2014、2016 | 相同 |  |
| `MCQ160G` | `HE_PFTdr` | 2014、2016 | 要轉換 | COPD 診斷 |
| `MCQ160K` | `HE_PFTdr` | 2014、2016 | 要轉換 | 同上 |
| `MCQ160L` | `DK4_dg / DK8_dg / DK9_dg` | 2014、2016 | 要轉換 | 肝硬化、B 肝、C 肝合併 |
| `liver_current` | `DK4_pr / DK8_pr / DK9_pr` | 2014、2016 | 要轉換 |  |
| `MCQ160M` | `DE2_dg` | 2014、2016 | 相同 |  |
| `thyroid_current` | `DE2_pr` | 2014、2016 | 相同 |  |
| `MCQ220` | `DC*_dg` | 2014、2016 | 要轉換 | 各癌別合併 |
| `MCQ300A` | `HE_IHDfh1–3` | 2014、2016 | 要轉換 | 缺血性心臟病家族史 |
| `MCQ300C` | `HE_DMfh1–3` | 2014、2016 | 要轉換 |  |
| `KIQ022` | `DN1_dg` | 2014、2016 | 要轉換 | 問「腎衰竭」 |
| `smoking_status` | `BS1_1 + BS3_1` | 2014、2016 | 要轉換 |  |
| `cigs_per_day` | `BS3_2` | 2014、2016 | 相同 |  |
| `home_smoke_inside` | `BS9_2` | 2014、2016 | 要轉換 | 問家中二手菸暴露 |
| `alcohol_freq` | `BD1_11` | 2014、2016 | 要轉換 | 類別不同 |
| `drinks_per_day` | `BD2_1` | 2014、2016 | 要轉換 | 以「杯」分類，標準杯不同 |
| `binge_freq` | `BD2_31` | 2014、2016 | 要轉換 | 門檻不同（男 7 杯 / 女 5 杯） |
| `sleep_hours` | `Total_slp_wk` | 只有 2016 | 要轉換 | 只有 2016；平日平均 |
| `hours_worked` | `EC_wht_23` | 2014、2016 | 要轉換 |  |
| `OCD150` | `EC1_1` | 2014、2016 | 要轉換 |  |
| `not_working_reason` | `EC1_2` | 2014、2016 | 要轉換 |  |
| `HOQ065` | `house` | 2014、2016 | 要轉換 |  |
| `WHQ030` | `BO1` | 2014、2016 | 要轉換 | 自覺體型 |
| `WHQ070` | `BO2_1` | 2014、2016 | 要轉換 |  |
| `WHD050` | `BO1_1–BO1_3` | 2014、2016 | 要轉換 | 只有體重變化方向和量 |
| `WHD080A` | `BO3_03` | 2014、2016 | 相同 |  |
| `WHD080D` | `BO3_01` | 2014、2016 | 相同 |  |
| `WHD080E` | `BO3_14` | 2014、2016 | 相同 |  |
| `WHD080I` | `BO3_04` | 2014、2016 | 相同 |  |
| `WHD080J` | `BO3_05 / BO3_07` | 2014、2016 | 要轉換 |  |
| `RHQ010` | `LW_ms_a` | 2014、2016 | 相同 |  |
| `RHQ031` | `LW_ms` | 2014、2016 | 要轉換 |  |
| `RHQ131` | `LW_pr` | 2014、2016 | 相同 |  |
| `RHQ160` | `LW_pr_1` | 2014、2016 | 相同 |  |
| `RHQ420` | `LW_oc` | 2014、2016 | 要轉換 |  |
| `PFQ049` | `LQ4_00` | 2014、2016 | 要轉換 | 活動受限 |
| `BMXBMI` | `HE_BMI` | 2014、2016 | 相同 |  |
| `BMXHT` | `HE_ht` | 2014、2016 | 相同 |  |
| `BMXWT` | `HE_wt` | 2014、2016 | 相同 |  |
| `BMXWAIST` | `HE_wc` | 2014、2016 | 相同 |  |
| `BPXSY1` | `HE_sbp1` | 2014、2016 | 要轉換 | 量測儀器與流程要核對 |
| `BPXSY2` | `HE_sbp2` | 2014、2016 | 要轉換 | 同上 |
| `BPXSY3` | `HE_sbp3` | 2014、2016 | 要轉換 | 同上 |
| `BPXDI1` | `HE_dbp1` | 2014、2016 | 要轉換 | 同上 |
| `BPXDI2` | `HE_dbp2` | 2014、2016 | 要轉換 | 同上 |
| `BPXDI3` | `HE_dbp3` | 2014、2016 | 要轉換 | 同上 |
| `BPXPLS` | `HE_mPLS` | 2014、2016 | 相同 |  |
| `BPXPULS` | `HE_rPLS` | 2014、2016 | 相同 |  |
| `LBXSASSI` | `HE_ast` | 2014、2016 | 相同 | 儀器不同 |
| `LBXSATSI` | `HE_alt` | 2014、2016 | 相同 | 儀器不同 |
| `LBXSBU` | `HE_BUN` | 2014、2016 | 相同 |  |
| `LBXSCR` | `HE_crea` | 2014、2016 | 相同 |  |
| `LBXSCH` | `HE_chol` | 2014、2016 | 要轉換 | KNHANES 是空腹檢體 |
| `LBXSGL` | `HE_glu` | 2014、2016 | 要轉換 | NHANES 生化盤非空腹，KNHANES 是空腹血糖 |
| `LBXSTR` | `HE_TG` | 2014、2016 | 要轉換 | 空腹對非空腹 |
| `LBXSUA` | `HE_Uacid` | 只有 2016 | 相同 | 只有 2016 |
| `URXUCR` | `HE_UCREA` | 2014、2016 | 相同 |  |
| `LBXTC` | `HE_chol` | 2014、2016 | 相同 |  |
| `LBDHDD` | `HE_HDL_st2` | 2014、2016 | 相同 | KNHANES 已換算 |
| `LBXGH` | `HE_HbA1c` | 2014、2016 | 相同 |  |
| `LBXCOT` | `HE_Ucot` | 2014、2016 | 要轉換 | 血清對尿液 cotinine |
| `LBXWBCSI` | `HE_WBC` | 2014、2016 | 相同 |  |
| `LBXRBCSI` | `HE_RBC` | 2014、2016 | 相同 |  |
| `LBXHGB` | `HE_HB` | 2014、2016 | 相同 |  |
| `LBXHCT` | `HE_HCT` | 2014、2016 | 相同 |  |
| `LBXMCVSI` | `HE_MCV` | 2014、2016 | 相同 |  |
| `LBXMCHSI` | `HE_MCH` | 2014、2016 | 相同 |  |
| `LBXMC` | `HE_MCHC` | 2014、2016 | 相同 |  |
| `LBXPLTSI` | `HE_Bplt` | 2014、2016 | 相同 |  |
| `DR_TKCAL` | `N_EN` | 2014、2016 | 要轉換 | 飲食都是 1 天回憶（NHANES 取 2 天平均），食物成分表不同 |
| `DR_TMOIS` | `N_WATER` | 2014、2016 | 要轉換 | 同上 |
| `DR_TPROT` | `N_PROT` | 2014、2016 | 要轉換 | 同上 |
| `DR_TTFAT` | `N_FAT` | 2014、2016 | 要轉換 | 同上 |
| `DR_TSFAT` | `N_SFA` | 2014、2016 | 要轉換 | 同上 |
| `DR_TMFAT` | `N_MUFA` | 2014、2016 | 要轉換 | 同上 |
| `DR_TPFAT` | `N_PUFA` | 2014、2016 | 要轉換 | 同上 |
| `DR_TCARB` | `N_CHO` | 2014、2016 | 要轉換 | 同上 |
| `DR_TCALC` | `N_CA` | 2014、2016 | 要轉換 | 同上 |
| `DR_TPHOS` | `N_PHOS` | 2014、2016 | 要轉換 | 同上 |
| `DR_TSODI` | `N_NA` | 2014、2016 | 要轉換 | 同上 |
| `DR_TPOTA` | `N_K` | 2014、2016 | 要轉換 | 同上 |
| `DR_TIRON` | `N_FE` | 2014、2016 | 要轉換 | 同上 |
| `DR_TBCAR` | `N_CAROT` | 2014、2016 | 要轉換 | 同上 |
| `DR_TRET` | `N_RETIN` | 2014、2016 | 要轉換 | 同上 |
| `DR_TVB1` | `N_B1` | 2014、2016 | 要轉換 | 同上 |
| `DR_TVB2` | `N_B2` | 2014、2016 | 要轉換 | 同上 |
| `DR_TNIAC` | `N_NIAC` | 2014、2016 | 要轉換 | 同上 |
| `DR_TVC` | `N_VITC` | 2014、2016 | 要轉換 | 同上 |
| `DR_TCHOL` | `N_CHOL` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TFIBE` | `N_TDF` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TSUGR` | `N_SUGAR` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TMAGN` | `N_MG` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TZINC` | `N_ZN` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TVARA` | `N_VA_RAE` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_TATOC` | `N_VITE` | 只有 2016 | 要轉換 | 只有 2016；α-TE 對 α-tocopherol |
| `DR_TFDFE` | `N_FOLATE` | 只有 2016 | 要轉換 | 只有 2016 |
| `DR_300` | `N_DUSUAL` | 2014、2016 | 相同 |  |
| `DRQSDIET` | `N_DIET` | 2014、2016 | 相同 |  |
| `DR_320Z` | `N_WAT_C` | 2014、2016 | 要轉換 | 以杯計 |

## 7. 下一步（待確認）

1. 決定以哪個資料集為主。建議 KNHANES：特徵重疊多（125 對 74），量表相同。
2. 下載 KNHANES 2018、2020（以及 2022，如果有）的基本資料檔和使用指南，確認 PHQ-9 是否都有。
3. 在 NHANES 上只用對得上的特徵跑 LOCO，和現有的全特徵結果比較，看精簡後損失多少。**KNHANES 已完成**（`modeling/step9_knhanes_features.ipynb`，11/12）：k125 的 AUROC 是 0.838（elastic net、XGBoost 相同），k114 是 0.834 / 0.833，比全部 317 個特徵低約 0.035。NHIS（PHQ-8 標籤）還沒做。
4. 逐項核對代碼並做轉換，在 NHANES 2005–18 全部資料上訓練，拿到 KNHANES（和 NHIS 2022）測試。報告 AUROC，並另外報告校準。
