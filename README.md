# personal_finance_tracker

Local backend workflow:

1. Start only Postgres:
```bash
docker compose up -d db
```

2. Run the FastAPI app on your machine with a host-local database URL:
```bash
export DATABASE_URL=postgresql+psycopg2://newuser:12345678@localhost:5433/tracker
alembic upgrade head
uvicorn api.main:app --reload
```

`docker-compose.yml` keeps the containerized `api` service pointed at `db`, while your host-run app should use `localhost`.
