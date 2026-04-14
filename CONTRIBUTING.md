# Contributing — Team Roles & Responsibilities

## Team Members

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Member 1 | ML Engineer | Data pipeline, model training, MLflow tracking, SHAP analysis |
| Member 2 | Backend Engineer | FastAPI implementation, schemas, model wrapper, Docker |
| Member 3 | DevOps Engineer | Prometheus, Grafana dashboards, alerting, CI/CD pipeline |
| Member 4 | QA / Documentation | Unit & integration tests, README, ARCHITECTURE.md, fairness analysis |

## Git Workflow

```
main          ← stable, production-ready
develop       ← integration branch
feature/*     ← individual features
```

### Branch naming
```
feature/data-pipeline
feature/model-training
feature/fastapi-endpoints
feature/monitoring-setup
feature/ci-cd
feature/responsible-ai
```

### Commit message format
```
feat: add SHAP explainability script
fix: handle zero account balance in preprocessing
test: add edge case tests for prediction endpoint
docs: update README with Docker setup instructions
```

## Development Setup

```bash
git clone <repo-url>
cd fraud-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=html
```

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] New code has test coverage
- [ ] No linting errors (`flake8 app/ tests/`)
- [ ] Updated README if needed
- [ ] Requested review from at least 1 team member