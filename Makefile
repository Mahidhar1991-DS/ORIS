.PHONY: bootstrap install test run extract docker-build docker-up docker-down clean

bootstrap:
	@set -e; \
	if ! command -v docker >/dev/null 2>&1; then \
		echo "Docker is required."; \
		exit 1; \
	fi; \
	if [ ! -f .env ]; then \
		printf "OpenAI API key: "; \
		read -s OPENAI_API_KEY; \
		echo; \
		if [ -z "$$OPENAI_API_KEY" ]; then \
			echo "OpenAI API key is required."; \
			exit 1; \
		fi; \
		printf '%s\n' \
			"OPENAI_API_KEY=$$OPENAI_API_KEY" \
			"OPENAI_MODEL=gpt-5.4-mini" \
			"MAX_USD_COST=0.10" \
			"PORT=8001" > .env; \
		echo "Created .env"; \
	else \
		echo ".env already exists"; \
	fi; \
	echo "Building ORIS..."; \
	docker compose build; \
	echo "Starting ORIS..."; \
	docker compose up -d; \
	echo "Waiting for ORIS..."; \
	for i in $$(seq 1 30); do \
		if curl -fsS http://127.0.0.1:8001/api/health >/dev/null 2>&1; then \
			echo "ORIS is ready: http://localhost:8001"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "ORIS did not become ready."; \
	docker compose logs --tail=50; \
	exit 1

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

run:
	uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

extract:
	@set -e; \
	if [ -z "$(LAT)" ] || [ -z "$(LON)" ] || [ -z "$(RADIUS_KM)" ]; then \
		echo 'Usage: make extract LAT="48.8566" LON="2.3522" RADIUS_KM="5"'; \
		exit 1; \
	fi; \
	if ! docker compose ps --status running --services | grep -q '^oris$$'; then \
		echo 'Starting ORIS with Docker...'; \
		docker compose up -d; \
	fi; \
	echo 'Submitting ORIS discovery job...'; \
	RESPONSE=$$(curl -fsS -X POST http://127.0.0.1:8001/api/jobs \
		-H 'Content-Type: application/json' \
		--data-binary '{"latitude":$(LAT),"longitude":$(LON),"radius_km":$(RADIUS_KM)}'); \
	JOB_ID=$$(printf '%s' "$$RESPONSE" | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'); \
	echo "Job ID: $$JOB_ID"; \
	mkdir -p results; \
	while true; do \
		RESPONSE=$$(curl -fsS http://127.0.0.1:8001/api/jobs/$$JOB_ID); \
		STATUS=$$(printf '%s' "$$RESPONSE" | python -c 'import json,sys; print(json.load(sys.stdin)["status"])'); \
		echo "Status: $$STATUS"; \
		if [ "$$STATUS" = "completed" ]; then \
			printf '%s\n' "$$RESPONSE" > results/latest.json; \
			echo 'Result saved to results/latest.json'; \
			break; \
		fi; \
		if [ "$$STATUS" = "failed" ]; then \
			printf '%s\n' "$$RESPONSE" > results/latest.json; \
			echo 'Job failed. Response saved to results/latest.json'; \
			exit 1; \
		fi; \
		sleep 2; \
	done; \
	cat results/latest.json

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
