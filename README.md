# FraudShield — Real-Time Credit Card Fraud Detection

---

# A. Problem Definition & Requirements 

---

## 1. Problem Statement

### 1.1. Bối cảnh kinh doanh (Business Context)

Gian lận thẻ tín dụng là một trong những vấn đề tốn kém nhất trong ngành tài chính. Theo **Nilson Report (2023)**, tổn thất do gian lận thẻ thanh toán toàn cầu đạt **$33.83 tỷ USD trong năm 2022** và dự kiến vượt $43 tỷ USD vào năm 2026. Theo **Federal Trade Commission (FTC) — Consumer Sentinel Network Data Book 2023**, gian lận thẻ tín dụng là loại hình identity theft phổ biến nhất tại Mỹ, chiếm hơn **440,000 báo cáo** trong năm 2023.

Hệ thống phát hiện gian lận truyền thống dựa trên quy tắc cứng (rule-based) ngày càng kém hiệu quả trước các phương thức gian lận tinh vi. Nghiên cứu từ **Association of Certified Fraud Examiners (ACFE, 2022)** cho thấy các tổ chức mất trung bình **5% doanh thu hàng năm** do gian lận, và thời gian trung bình để phát hiện gian lận là **12 tháng**.

Hiện tại, nhiều tổ chức tài chính vẫn dựa vào **hệ thống rule-based** (ví dụ: chặn giao dịch trên $10,000 từ quốc gia lạ) để phát hiện gian lận. Cách tiếp cận này có những hạn chế rõ ràng:

- **Tỷ lệ False Positive cao** (~30-50%): Quá nhiều giao dịch hợp lệ bị chặn nhầm, gây trải nghiệm xấu cho khách hàng và tốn chi phí nhân sự review thủ công.
- **Không thích ứng được với các pattern gian lận mới:** Kẻ gian lận liên tục thay đổi hành vi (concept drift), trong khi rules cố định cần thời gian dài để cập nhật.
- **Không thể scale:** Khi lượng giao dịch tăng, việc duy trì và mở rộng hàng nghìn rules trở nên bất khả thi.

### 1.2. Bài toán cần giải quyết

Xây dựng hệ thống phát hiện gian lận giao dịch thẻ tín dụng end-to-end dựa trên Machine Learning, sử dụng **Credit Card Transactions Fraud Detection Dataset** (Kaggle — kartik2112). Hệ thống phải có khả năng:

- Phân loại giao dịch thành **legitimate “Hợp lệ” (0)** hoặc **fraud “gian lận” (1)** một cách tự động dựa trên đặc trưng giao dịch và hành vi người dùng
- Phục vụ prediction qua REST API.
- Giám sát hiệu suất model và hệ thống liên tục thông qua metrics và dashboards
- Đảm bảo tính công bằng và giải thích được quyết định để tuân thủ các quy định tài chính

### 1.3. Tại sao Machine Learning?

| Phương pháp | Hạn chế |
| --- | --- |
| Rule-based (ngưỡng cố định) | Không thích ứng với mẫu gian lận mới; tỷ lệ false positive cao |
| Manual review | Không mở rộng được; tốn nhân lực; chậm |
| **ML-based (đề xuất)** | Học từ dữ liệu lịch sử; phát hiện mẫu phức tạp; tự động và mở rộng được |

---

## 2. Dataset

### 2.1. Tổng quan

| Thuộc tính | Chi tiết |
| --- | --- |
| **Nguồn** | [Kaggle — Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data) |
| **Phương pháp tạo** | Simulated bằng **Sparkov Data Generation** tool, mô phỏng giao dịch thực tế |
| **Thời gian** | Tháng 1/2019 — Tháng 12/2020 |
| **Quy mô** | ~1,000 khách hàng, ~800 merchants |
| **Tệp dữ liệu** | `fraudTrain.csv` (training) và `fraudTest.csv` (testing) |
| **Target variable** | `is_fraud` (0 = Legitimate, 1 = Fraud) |
| **Loại bài toán** | Binary Classification — **Highly Imbalanced** |

### 2.2. Các features chính

| Nhóm | Features | Ý nghĩa cho fraud detection |
| --- | --- | --- |
| **Giao dịch** | `amt`, `trans_date_trans_time`, `unix_time`, `trans_num` | Giá trị giao dịch bất thường, thời điểm giao dịch khả nghi |
| **Merchant** | `merchant`, `category` | Loại merchant rủi ro cao (vd: online shopping, grocery) |
| **Địa lý — Khách hàng** | `lat`, `long`, `city`, `state`, `zip`, `city_pop` | Vị trí khách hàng |
| **Địa lý — Merchant** | `merch_lat`, `merch_long` | Khoảng cách giữa khách hàng và merchant (feature engineering) |
| **Nhân khẩu học** | `gender`, `dob`, `job` | Phân tích fairness; tuổi có thể liên quan đến rủi ro |
| **Định danh** | `cc_num`, `first`, `last`, `street` | Cần loại bỏ hoặc mã hóa — không dùng làm feature do privacy |

### 2.3. Đặc điểm quan trọng của dataset

**Class imbalance:** Dataset có tỷ lệ fraud rất thấp so với legitimate transactions (thường khoảng **0.5–2% fraud**). Đây là đặc điểm phổ biến trong fraud detection thực tế — theo nghiên cứu của **Dal Pozzolo et al. (2015), "Calibrating Probability with Undersampling for Unbalanced Classification"**, datasets gian lận thực tế thường có tỷ lệ fraud dưới 1%.

**Dữ liệu mô phỏng nhưng realistic:** Dataset được sinh bằng Sparkov Data Generation dựa trên phân phối giao dịch thực tế, bao gồm seasonality, biến động theo giờ trong ngày, và các mẫu chi tiêu đa dạng theo category.

---

## 3. User Requirements & Use Cases

### 3.1. Stakeholders

| Stakeholder | Vai trò | Nhu cầu chính |
| --- | --- | --- |
| **Fraud Analyst** | Xem xét giao dịch bị gắn cờ | Danh sách giao dịch khả nghi, lý do model đánh dấu (explainability) |
| **Risk Manager** | Giám sát tổng thể hiệu quả hệ thống | Metrics dashboard, báo cáo fraud rate, false positive rate |
| **IT/DevOps Engineer** | Vận hành hệ thống | Container deployment, health monitoring, alerting |
| **Data Scientist** | Cải thiện model | Experiment tracking, model comparison, retraining pipeline |
| **Khách hàng (gián tiếp)** | Không bị block giao dịch hợp lệ | Tỷ lệ false positive thấp |

### 3.2. Use Cases

**UC-01: Phân loại giao dịch**

- *Actor:* Hệ thống thanh toán
- *Mô tả:* Khi một giao dịch mới được thực hiện, hệ thống gửi thông tin giao dịch tới FraudShield API. Hệ thống trả về nhãn dự đoán (Normal/Fraud) cùng điểm xác suất gian lận (fraud probability score).
- *Flow:* Giao dịch mới → Gửi tới FraudShield API → Nhận prediction (is_fraud: 0/1) + fraud_probability (0–1)
- *Input:* amt, category, merchant, lat/long, trans_date_trans_time, gender, dob, v.v.
- *Output:* `{"is_fraud": 0, "fraud_probability": 0.03, "explanation": {...}}`

**UC-02:**  **Giám sát hiệu suất model**   

- *Actor:* Risk Manager, Data Scientist
- *Mô tả:* Theo dõi hiệu suất model qua thời gian (precision, recall, F1) thông qua Grafana dashboard; nhận cảnh báo khi model performance giảm.
- *Flow:* Truy cập Grafana dashboard → Xem precision, recall, F1 theo thời gian → Nhận alert nếu metrics giảm

**UC-03: Retrain model khi cần**

- *Actor:* Data Scientist
- *Mô tả:* Khi phát hiện model degradation hoặc có dữ liệu mới, Data Scientist trigger retraining pipeline, so sánh model mới với model hiện tại qua MLflow, và quyết định deploy model mới.
- *Flow:* Phát hiện performance degradation → Trigger retraining → So sánh model mới vs. hiện tại trên MLflow → Deploy nếu cải thiện

**UC-04: Giải thích quyết định**

- *Actor:* Fraud Analyst
- *Flow:* Chọn giao dịch bị gắn cờ → Xem giải thích model  để hiểu tại sao giao dịch bị đánh dấu → Quyết định confirm hoặc dismiss

### 3.3. Functional Requirements

| ID | Requirement | Priority | Giải thích |
| --- | --- | --- | --- |
| FR-01 | REST API nhận thông tin giao dịch, trả về prediction + probability | **Must-have** | Core functionality |
| FR-02 | ML Pipeline: data preprocessing → feature engineering → model training → evaluation | **Must-have** | Yêu cầu đồ án |
| FR-03 | Experiment tracking qua MLflow (log metrics, params, model artifacts) | **Must-have** | Yêu cầu đồ án |
| FR-04 | Docker containerization cho tất cả services | **Must-have** | Yêu cầu đồ án |
| FR-05 | Docker Compose orchestration cho multi-service deployment | **Must-have** | Yêu cầu đồ án |
| FR-06 | Prometheus metrics collection + Grafana dashboards | **Must-have** | Yêu cầu đồ án |
| FR-07 | Alerting rules khi model metrics giảm dưới ngưỡng | **Should-have** | Best practice |
| FR-08 | Model explainability (SHAP) cho từng prediction | **Should-have** | Responsible AI |
| FR-09 | Health check endpoint | **Must-have** | Production readiness |
| FR-10 | API documentation (Swagger/OpenAPI) | **Must-have** | Yêu cầu đồ án |
| FR-11 | CI/CD pipeline qua GitHub Actions | **Must-have** | Yêu cầu đồ án |

---

## 4. Success Metrics

### 4.1. Model-Level Metrics

| Metric | Target | Cơ sở |
| --- | --- | --- |
| **AUC-ROC** | ≥ 0.90 | Dựa trên kết quả public trên Kaggle cho dataset này — nhiều submissions đạt AUC 0.90–0.99 trên leaderboard |
| **Recall (Fraud)** | ≥ 0.85 | Trong fraud detection, recall quan trọng hơn precision vì bỏ lọt gian lận tốn kém hơn false alarm. Con số 0.80 là baseline hợp lý — nghiên cứu của **Bhattacharyya et al. (2011), "Data mining for credit card fraud: A comparative study"** cho thấy các model tốt đạt recall 0.75–0.90 |
| **Precision (Fraud)** | ≥ 0.50 | Chấp nhận một mức false positive nhất định để đảm bảo recall cao. Trên thực tế, nhiều hệ thống chấp nhận precision thấp hơn recall |
| **F1-Score (Fraud)** | ≥ 0.60 | Cân bằng giữa precision và recall |
| **PR-AUC** | ≥ 0.70 | Metric phù hợp hơn AUC-ROC cho imbalanced data — theo **Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** |

**Ghi chú:** Các target trên là do nhóm đặt ra dựa trên benchmark công khai của dataset này trên Kaggle và các nghiên cứu liên quan. Sau khi train model thực tế, nhóm sẽ điều chỉnh nếu cần.

### 4.2. System-Level Metrics

| Metric | Target | Cơ sở |
| --- | --- | --- |
| API Latency (P95) | < 500ms | Google SRE guidelines cho user-facing services |
| API Throughput | ≥ 50 req/min | Hợp lý cho demo trên local Docker |
| Container Health | 100% services healthy | Docker Compose health checks |
| CI/CD Pipeline Pass Rate | ≥ 95% | Industry standard |

### 4.3. Business-Level Metrics

| Metric | Mô tả | Cách đo |
| --- | --- | --- |
| **Fraud Detection Rate** | % giao dịch gian lận được phát hiện | Recall on test set |
| **Customer Impact Rate** | % giao dịch hợp lệ bị block nhầm | False Positive Rate on test set |
| **Estimated Savings** | Tổng `amt` của fraud transactions được detect đúng | Sum of `amt` where TP |

---

## 5. Scope Definition & Constraints

### 5.1. Trong phạm vi (In Scope)

- ML pipeline hoàn chỉnh: data ingestion → EDA → preprocessing → feature engineering → model training → evaluation
- Feature engineering từ raw data (khoảng cách customer-merchant, giờ trong ngày, ngày trong tuần, tuổi khách hàng, v.v.)
- Xử lý class imbalance (SMOTE, class weights, hoặc threshold tuning)
- REST API serving prediction (FastAPI)
- Docker + Docker Compose deployment
- Prometheus + Grafana monitoring
- MLflow experiment tracking
- GitHub Actions CI/CD
- Responsible AI: fairness analysis theo gender, explainability qua SHAP
- Testing: unit, integration, data quality, model validation

### 5.2. Ngoài phạm vi (Out of Scope)

- Tích hợp với hệ thống thanh toán thực tế
- Real-time streaming (Kafka/Flink)
- Cloud deployment (AWS/GCP/Azure)
- Frontend application
- Xử lý PII (Personally Identifiable Information) theo GDPR/CCPA compliance đầy đủ — chỉ thực hiện ở mức loại bỏ PII khỏi features

### 5.3. Constraints

| Loại | Mô tả |
| --- | --- |
| **Dữ liệu** | Dataset simulated (không phải dữ liệu thực) — kết quả model có thể không phản ánh chính xác hiệu suất trên dữ liệu thực |
| **Class imbalance** | Fraud chiếm tỷ lệ rất nhỏ (~0.5–2%) — cần kỹ thuật xử lý đặc biệt |
| **PII trong dataset** | Dataset chứa tên, địa chỉ, credit card number — cần loại bỏ khỏi model features và không log vào monitoring |
| **Thời gian** | 1 tuần phát triển |
| **Hạ tầng** | Local Docker environment, không có GPU |
| **Đội ngũ** | 4 thành viên |

### 5.4. Assumptions

- Nhãn `is_fraud` trong dataset là chính xác (ground truth)
- Phân phối giao dịch trong dataset (mặc dù simulated) đủ realistic để train model có ý nghĩa
- Fraud patterns trong training set cũng xuất hiện trong test set (không có concept drift giữa train/test)

### 5.5. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
| --- | --- | --- | --- |
| **Class imbalance** dẫn đến model predict toàn Normal | Cao | Cao | SMOTE, undersampling, class_weight='balanced', threshold tuning; đánh giá bằng PR-AUC thay vì accuracy |
| **Overfitting** trên training data | Trung bình | Trung bình | Cross-validation, regularization, đánh giá trên test set riêng |
| **Data leakage** từ features có tương quan trực tiếp với target | Cao | Thấp | Review feature engineering cẩn thận; không dùng features tạo sau giao dịch |
| **PII exposure** trong logs/API | Cao | Thấp | Input sanitization, không log sensitive fields, loại bỏ PII khỏi model |
| **Model bias** theo gender hoặc vùng miền | Trung bình | Trung bình | Fairness analysis, equalized odds check, SHAP analysis per group |

---

## Tổng kết nguồn tham khảo

| Nguồn | Nội dung sử dụng |
| --- | --- |
| **Nilson Report (2023)** | Tổn thất gian lận thẻ toàn cầu $33.83B (2022) |
| **FTC Consumer Sentinel Network (2023)** | 440,000+ báo cáo fraud thẻ tín dụng tại Mỹ |
| **ACFE Report to the Nations (2022)** | Tổ chức mất ~5% doanh thu do fraud; median detection time 12 tháng |
| **Bhattacharyya et al. (2011)** | Benchmark recall 0.75–0.90 cho fraud detection models |
| **Saito & Rehmsmeier (2015)** | PR-AUC phù hợp hơn ROC-AUC cho imbalanced data |
| **Dal Pozzolo et al. (2015)** | Tỷ lệ fraud thực tế thường < 1% |
| **Kaggle public leaderboard** | AUC-ROC 0.95–0.99 cho dataset kartik2112 |
| **Google SRE Book** | API latency target < 500ms P95 |
|  |  |

![CI](https://github.com/your-org/fraud-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Model](https://img.shields.io/badge/model-GBM-orange)
![Docker](https://img.shields.io/badge/docker-ready-blue)

An end-to-end ML system for real-time credit card fraud detection. Built with scikit-learn GBM, FastAPI, MLflow, Prometheus, and Grafana.

**Model Performance:**
- PR-AUC: **0.821** | F1: **0.784** | Precision: **0.847** | Recall: **0.730**

## Features

- 🔍 **Real-time fraud scoring** with probability and risk level (LOW/MEDIUM/HIGH/CRITICAL)
- 🤖 **GBM model** with manual oversampling — PR-AUC 0.82 on 555K test transactions
- 🎯 **Threshold tuning** — optimal F1 threshold found automatically
- 🔬 **Model comparison** — LogisticRegression, RandomForest, GBM benchmarked
- 📊 **MLflow experiment tracking** — local (`mlruns/`)
- 📈 **Prometheus + Grafana** monitoring with fraud-specific dashboards
- 🧠 **SHAP explainability** + **Fairness analysis** (gender, category, state)
- 🐳 **Fully containerized** with Docker Compose
- 🔄 **GitHub Actions CI/CD**

## Project Structure

```
fraud-detection/
├── app/
│   ├── main.py               # FastAPI application
│   ├── model.py              # Model wrapper + feature engineering
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── metrics.py            # Prometheus metrics definitions
│   ├── middleware.py         # HTTP metrics middleware
│   └── config.py             # Configuration
├── scripts/
│   ├── train_model.py        # Compare, tune & save best model
│   ├── evaluate_model.py     # Evaluation plots
│   ├── responsible_ai.py     # SHAP + fairness analysis
│   └── load_test.py          # Load testing
├── tests/
│   ├── test_api.py           # API integration tests
│   ├── test_model.py         # Model unit tests
│   └── test_data.py          # Data quality tests
├── data/
│   └── raw/                  # fraudTrain.csv + fraudTest.csv
├── models/                   # Saved model files (auto-generated)
├── reports/                  # Evaluation plots (auto-generated)
├── prometheus/               # Prometheus config + alert rules
├── grafana/                  # Grafana dashboards + provisioning
├── .github/workflows/        # GitHub Actions CI/CD
├── docker-compose.yml
├── Dockerfile
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

## Prerequisites

- Python 3.11
- Docker + Docker Compose

## Installation

```bash
git clone <your-repo-url>
cd fraud-detection

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download from Kaggle:
**https://www.kaggle.com/datasets/kartik2112/fraud-detection/data**

Place both files at:
```
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

Dataset info:
- **Train:** ~1.3M transactions | Fraud rate: 0.58%
- **Test:**  ~555K transactions | Fraud rate: 0.39%
- **Features:** transaction amount, merchant category, location coordinates, cardholder demographics
- **Target:** `is_fraud` (0 = normal, 1 = fraud)

## ML Pipeline

```bash
# Train model (compare 3 models → save best by PR-AUC)
python scripts/train_model.py

# Evaluate
python scripts/evaluate_model.py
# → Saves reports/confusion_matrix.png, roc_curve.png, etc.

# Responsible AI
python scripts/responsible_ai.py
# → Saves reports/shap_importance.png, fairness_analysis.png
```

> No separate preprocess step — feature engineering runs automatically inside `train_model.py`.

## Prediction Flow

```
API Request (JSON)
        ↓
app/model.py._preprocess()
  ├── Extract hour, day_of_week, month, is_night, is_weekend from trans_date_trans_time
  ├── Calculate age from dob
  ├── Calculate distance (haversine: cardholder lat/long → merchant lat/long)
  ├── Compute amt_vs_category_mean ratio
  ├── LabelEncode category and state (using encoders from pipeline.pkl)
  └── Encode gender (M=1, F=0)
        ↓
numpy array (13 features)
        ↓
fraud_model.pkl.predict_proba()
        ↓
fraud_probability = 0.92
        ↓
probability >= best_threshold (from pipeline.pkl → 0.851)
        ↓
is_fraud = True → risk_level = "CRITICAL"
        ↓
JSON Response
```

> `pipeline.pkl` and `fraud_model.pkl` must always be trained together.

## Deployment

```bash
docker-compose up --build -d
```

| Service    | URL                        | Credentials |
|------------|----------------------------|-------------|
| API        | http://localhost:8000      | —           |
| API Docs   | http://localhost:8000/docs | —           |
| Prometheus | http://localhost:9090      | —           |
| Grafana    | http://localhost:3000      | admin/admin |

```bash
# Train model inside container
docker-compose exec api python scripts/train_model.py

docker-compose up -d              # Start all
docker-compose down               # Stop all
docker-compose logs -f api        # View logs
docker-compose ps                 # Check status
```

### Test Cases

**🔴 Fraud — late night, high amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-02 01:06:00",
    "amt": 281.06,
    "category": "grocery_pos",
    "gender": "M",
    "city_pop": 885,
    "dob": "15/9/88",
    "lat": 35.9946,
    "long": -81.7266,
    "merch_lat": 36.430,
    "merch_long": -81.179,
    "state": "NC"
  }'
```

**🔴 Fraud — 3am, unusual amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-02 03:05:00",
    "amt": 276.31,
    "category": "grocery_pos",
    "gender": "F",
    "city_pop": 1595797,
    "dob": "28/10/60",
    "lat": 29.44,
    "long": -98.459,
    "merch_lat": 29.273,
    "merch_long": -98.836,
    "state": "TX"
  }'
```

**🟢 Normal — daytime, small amount**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-01 14:30:00",
    "amt": 4.97,
    "category": "misc_net",
    "gender": "F",
    "city_pop": 3495,
    "dob": "9/3/88",
    "lat": 36.0788,
    "long": -81.1781,
    "merch_lat": 36.011,
    "merch_long": -82.048,
    "state": "NC"
  }'
```

**🟢 Normal — business hours grocery**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-01 12:14:00",
    "amt": 29.84,
    "category": "personal_care",
    "gender": "F",
    "city_pop": 302,
    "dob": "17/1/90",
    "lat": 40.320,
    "long": -110.436,
    "merch_lat": 39.450,
    "merch_long": -109.960,
    "state": "UT"
  }'
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true, "model_version": "1.0.0"}
```

### Predict Fraud

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-06-21 12:14:00",
    "amt": 1500.00,
    "category": "shopping_net",
    "gender": "M",
    "city_pop": 333497,
    "dob": "19/3/68",
    "lat": 33.986,
    "long": -81.200,
    "merch_lat": 34.421,
    "merch_long": -82.611,
    "state": "SC"
  }'
```
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9231,
  "risk_level": "CRITICAL",
  "model_version": "1.0.0",
  "latency_ms": 8.3
}
```

### Python Example

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "trans_date_trans_time": "2019-06-21 14:30:00",
    "amt": 29.84,
    "category": "personal_care",
    "gender": "F",
    "city_pop": 302,
    "dob": "17/1/90",
    "lat": 40.320,
    "long": -110.436,
    "merch_lat": 39.450,
    "merch_long": -109.960,
    "state": "UT"
})
print(response.json())
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=app --cov-report=html

# By file
pytest tests/test_api.py -v      # API integration tests (mock model)
pytest tests/test_model.py -v    # Feature engineering + risk level unit tests
pytest tests/test_data.py -v     # Data quality tests (requires dataset in data/raw/)
```

### Test Coverage

| File | What it tests |
|------|--------------|
| `test_api.py` | All endpoints (health, predict, model/info, metrics), validation errors (422), 503 when model not loaded, all 14 categories, both genders |
| `test_model.py` | `engineer_features()` — hour, is_night, is_weekend, age, distance, gender_enc, amt_vs_category_mean; haversine distance; risk level (LOW/MEDIUM/HIGH/CRITICAL) |
| `test_data.py` | Schema, target binary, fraud rate range, no excessive nulls, date/dob parseable, engineered feature validity |

> `test_api.py` uses mock model — no `.pkl` file needed. `test_data.py` requires `data/raw/fraudTrain.csv` and `fraudTest.csv`.

### Sample Test Cases

| Case | Transaction | Expected |
|------|------------|----------|
| Normal | $4.97 misc_net, 2pm NC | `is_fraud=false`, LOW |
| Fraud | $281.06 grocery, 1am NC | `is_fraud=true`, HIGH/CRITICAL |
| Edge | $9999.99 any category | 200 OK |
| Edge | 3am transaction | 200 OK |
| Invalid | Missing `amt` field | 422 |
| Invalid | Negative `amt` | 422 |

## Load Testing

```bash
python scripts/load_test.py --duration 60 --workers 10
python scripts/load_test.py --spike
```

## Responsible AI

```bash
python scripts/responsible_ai.py
```

Generates in `reports/`:
- `shap_importance.png` — Top features driving fraud predictions
- `shap_beeswarm.png` — Feature impact distribution
- `fairness_analysis.png` — Recall by gender, category, state

## Monitoring

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_predictions_total` | Counter | Predictions by result (fraud/normal) |
| `fraud_detection_rate` | Gauge | Rolling fraud rate |
| `fraud_probability_distribution` | Histogram | Score distribution |
| `high_risk_transactions_total` | Counter | HIGH/CRITICAL risk count |
| `fraud_prediction_duration_seconds` | Histogram | Inference latency |
| `http_requests_total` | Counter | API request count |

### Grafana Dashboard (`http://localhost:3000`)

The dashboard has 12 panels:

| Panel | Type | Query | What it shows |
|-------|------|-------|---------------|
| **Model Status** | Stat | `ml_model_loaded` | Green = model loaded, Red = model down |
| **Fraud Rate (Rolling)** | Gauge | `fraud_detection_rate * 100` | % of recent 1000 predictions flagged as fraud. Threshold: yellow >5%, red >15% |
| **Prediction Rate** | Stat | `rate(fraud_predictions_total[5m])` | How many predictions per second |
| **HIGH/CRITICAL Alerts** | Stat | `rate(high_risk_transactions_total[5m])` | Rate of high-risk transactions being detected |
| **API Error Rate** | Stat | `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100` | % of requests returning 5xx errors |
| **P95 Prediction Latency** | Stat | `histogram_quantile(0.95, rate(fraud_prediction_duration_seconds_bucket[5m])) * 1000` | 95th percentile model inference time in ms |
| **Predictions Over Time** | Time series | `rate(fraud_predictions_total[5m])` by result | Normal vs Fraud prediction volume over time |
| **Fraud Probability Distribution** | Time series | `histogram_quantile(0.50/0.95/0.99, ...)` | Spread of fraud probability scores — P50, P95, P99 |
| **Request Rate by Endpoint** | Time series | `rate(http_requests_total[5m])` | Traffic per endpoint (predict, health, metrics) |
| **API Latency P50/P95/P99** | Time series | `histogram_quantile(...)` | End-to-end HTTP response time percentiles |
| **High Risk Transactions Over Time** | Time series | `rate(high_risk_transactions_total[5m])` by risk_level | HIGH vs CRITICAL risk transactions over time |
| **Status Code Distribution** | Pie chart | `sum by (status) (rate(http_requests_total[5m]))` | Breakdown of 200, 422, 503 responses |

### Alert Rules (8 total)

View active alerts at `http://localhost:9090/alerts`

| Alert | Condition | Severity | Meaning |
|-------|-----------|----------|---------|
| `ModelNotLoaded` | `ml_model_loaded == 0` for 30s | Critical | Model failed to load — all predictions returning 503 |
| `HighFraudRate` | `fraud_detection_rate > 0.15` for 2m | Critical | More than 15% of recent transactions flagged as fraud — possible attack |
| `HighPredictionLatency` | P95 > 100ms for 2m | Warning | Model inference too slow |
| `HighRiskTransactionSpike` | `rate(high_risk_transactions_total[5m]) > 5` for 1m | Warning | Sudden spike in HIGH/CRITICAL risk detections |
| `PredictionErrors` | `rate(fraud_prediction_errors_total[5m]) > 0.1` for 1m | Critical | Model throwing errors on valid requests |
| `HighErrorRate` | HTTP 5xx > 10% for 1m | Critical | Too many server errors |
| `HighRequestLatency` | HTTP P95 > 1s for 2m | Warning | API responding too slowly |
| `APIDown` | `up{job="fraudshield-api"} == 0` for 30s | Critical | API container unreachable |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not loaded` (503) | Run `python scripts/train_model.py` first |
| Dataset not found | Download from Kaggle link above, place in `data/raw/` |
| Grafana no data | Check `localhost:9090/targets` — API must be UP |
| Docker build fails | Use `python:3.10` (not `slim`) in Dockerfile |
| numpy compatibility | Ensure `numpy<2` in requirements.txt |