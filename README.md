# ORIS — Quarry Intelligence & Research System

ORIS is a small pipeline for finding quarry sites around a given location and collecting useful information about them.

The input is a latitude, longitude, and search radius. ORIS first looks for quarry sites using OpenStreetMap/Overpass. It then tries to find an official website for each site. When a website is available, the crawler collects the public content and the extraction step turns that content into structured quarry information.

The extracted information is stored together with evidence and source information so that the result can be checked later.

---

## 1. Architecture & Data Flow

```text
User / Browser
      |
      v
   FastAPI
      |
      v
 Job Queue / Worker
      |
      v
OpenStreetMap / Overpass
      |
      v
 Quarry Discovery
      |
      v
Municipality Enrichment
      |
      v
Website Discovery
      |
      v
 Web Crawler
      |
      v
 LLM Extraction
      |
      +------------------+
      |                  |
      v                  v
Structured Data     Token / Cost
      |
      v
Evidence & Sources
      |
      v
Location Verification
      |
      v
    SQLite
      |
      v
API / Browser
```

The main flow is:

1. The user provides coordinates and a radius.
2. FastAPI creates a job and puts it on the worker queue.
3. The worker runs the discovery pipeline.
4. OpenStreetMap/Overpass is used to find possible quarry sites.
5. Municipality information is added to the discovered sites.
6. ORIS tries to find an official website for each site.
7. If a website is available, the crawler collects its public content.
8. The LLM extracts the fields defined by the ORIS data model.
9. Evidence and source information are kept with the extracted fields.
10. Location verification is handled separately from the LLM.
11. Token usage and estimated cost are recorded.
12. The final result is saved in SQLite and returned through the API.

---

## 2. Design Decisions & Trade-offs

### OpenStreetMap / Overpass

I used OpenStreetMap as the starting point because the first part of the problem is geographic discovery. Overpass makes it possible to query the relevant area without maintaining a separate geographic dataset.

The main limitation is that OSM data is not always complete. A quarry may be present but have limited information, or may not be tagged in a way that the query can find.

### Website discovery

I kept website discovery separate from extraction. This is important because finding a website and deciding what information is on that website are two different problems.

If ORIS cannot establish an official website with enough confidence, the site is returned as `no_website` instead of trying to extract information from an uncertain source.

This means some sites will stop at this stage, but it avoids presenting information from the wrong company or website.

### LLM extraction

The LLM is used after the source content has been collected. Its job is to turn the website content into the structured fields required by ORIS.

I also keep evidence with the extracted values. This makes it possible to see where an extracted value came from instead of treating the LLM output as the source itself.

The trade-off is that LLM calls add API cost and latency. For this reason the pipeline tracks token usage and estimated USD cost and also supports an optional per-run budget.

### Location verification

Location verification is handled separately from the LLM.

I preferred this approach because geographic checks should be deterministic where possible rather than relying on an LLM to decide whether coordinates match a location.

### SQLite and the worker

I used SQLite and an in-process asynchronous worker to keep the project easy to run locally and with Docker.

For this assignment that keeps the setup small: there is no separate database or queue that the reviewer has to install.

The trade-off is that this would not be the architecture I would use for a large production workload. In that case I would move the data to PostgreSQL and use a durable queue with separate worker processes.

---

## 3. Running the Project

### Requirements

The project can be run with Docker. You will need:

- Docker
- Docker Compose
- An OpenAI API key for the LLM extraction step

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

The `.env` file should not be committed to the repository.

### Start the project

Build the image:

```bash
make bootstrap
```

Start ORIS:

```bash
make docker-up
```

Check that the API is running:

```bash
curl http://127.0.0.1:8001/api/health
```

Expected response:

```json
{"status":"ok"}
```

The UI is available at:

```
http://127.0.0.1:8001
```

### Run a discovery

The main command for running a search is:

```bash
make extract LAT="48.9362" LON="2.3165" RADIUS_KM="5"
```

The command submits the job, waits for it to finish, and saves the response to:

```
results/latest.json
```

For example:

```text
Submitting ORIS discovery job...
Job ID: 9b3589ac-6d80-4fcc-8dab-13ef3c6937c6
Status: running
Status: completed
Result saved to results/latest.json
```

For the coordinates above, the test run found a quarry site in Gennevilliers. No official website could be established for that site, so the result was:

```
status: no_website
```

That is an expected result from the pipeline, not an application error.

### Run the tests

```bash
python -m pytest -q
```

Current result:

```
64 passed, 5 warnings
```

The warnings are dependency-level SWIG/PyMuPDF deprecation warnings. They do not cause any tests to fail.

---

## 4. Evaluation

I included a small ground-truth example under:

```
eval/ground_truth.json
```

The scoring code is:

```
eval/evaluate.py
```

and the corresponding extraction result is:

```
eval/tersen_result.json
```

The example uses a known Tersen quarry source and contains the expected values for the fields that ORIS extracts.

Run the evaluation with:

```bash
python eval/evaluate.py eval/tersen_result.json
```

### What is evaluated

The script compares the extracted result with the ground truth for:

- Official name
- Site type
- Description
- Operational status
- Materials produced
- Certifications
- Location verification
- Extracted city

It also checks the evidence attached to extracted fields.

### Metrics

The first metric is **field accuracy**:

```
correct fields / all evaluated fields
```

The second metric is the **evidence score**:

```
fields with valid evidence / fields requiring evidence
```

### Current result

The current evaluation gives:

```text
ORIS EVALUATION
==================================================
Accuracy:       100.0%
Evidence score: 100.0%

PASS  official_name
PASS  site_type
PASS  description
PASS  operational_status
PASS  materials_produced
PASS  certifications
PASS  location_verified
PASS  location_city

Evidence: 2/2
```

So the current test case has:

- 8/8 fields passing
- 100% field accuracy
- 100% evidence score
- 2/2 evidence checks passing

This is a small ground-truth set, so I would not consider these numbers to be a measure of real-world production accuracy. A larger set would be needed to evaluate the system across different quarry sites, websites, languages, and missing-data cases.

---
