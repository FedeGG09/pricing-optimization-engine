# Pricing Optimization Engine

<img width="1631" height="784" alt="image" src="https://github.com/user-attachments/assets/fca180bc-aadf-4aab-bc8d-74478bb14f17" />

Interactive Demo:  # https://argent-pricing.lovable.app/

An industrial pricing demo built to showcase **data-driven pricing decisions, customer segmentation, commercial intelligence, and AI-assisted explanation workflows**.

This project simulates a real pricing engine for industrial sales teams. It combines a FastAPI backend, a lightweight data layer, a rule-based pricing engine, and a polished frontend experience designed for executive demos and portfolio presentations.

<img width="1902" height="880" alt="image" src="https://github.com/user-attachments/assets/77e3f5cc-e230-4902-b714-c9c3c9fd054d" />


---

## Why this project exists

Pricing is one of the most impactful levers in industrial sales.  
This application was built to demonstrate how commercial teams can:

- recommend prices with business rules and historical context
- analyze customer behavior and commercial risk
- compare pricing decisions against market pressure
- explain recommendations in a clear, executive-friendly way
- simulate AI-assisted storytelling without relying on unstable external services

The goal is not just to show data — it is to show **decision support**.

---

## Key capabilities

- **Customer and product selection**
- **Pricing recommendation simulation**
- **Commercial explanation layer**
- **What-if scenario analysis**
- **Customer history and commercial context**
- **Competitive pressure signals**
- **AI-style commercial narration**
- **Audit trail and model metrics**
- **Dark premium UI for demo and stakeholder presentations**

---

## Tech stack

### Backend
- **FastAPI** — API layer and business endpoints
- **Python** — core application logic
- **SQLite** — lightweight local persistence
- **Pandas** — data processing and feature preparation
- **scikit-learn** — modeling utilities and scoring logic
- **joblib** — model serialization
- **Pydantic** — typed request/response schemas
- **python-dotenv** — environment variable loading

### Frontend
- **HTML**
- **CSS**
- **JavaScript**
- Responsive dashboard layout
- Custom dark UI for executive-facing demos

### AI / Narrative layer
- Simulated LLM-style responses
- Programmed commercial narratives
- Rule-based explanations and scenario outputs
- Hugging Face-ready integration layer for future extension

<img width="1878" height="874" alt="image" src="https://github.com/user-attachments/assets/cf2d90d4-7ed0-44e0-8b28-f2caa4cf68f0" />


### Deployment
- Designed for modern cloud deployment
- Compatible with static frontend hosting and serverless/backend hosting patterns

---

## What the app does

The system simulates an industrial pricing workflow:

1. A user logs in to the demo.
2. The dashboard loads customer, product, and regional context.
3. The engine generates a pricing recommendation.
4. The app explains why that price was suggested.
5. The user can test alternative scenarios.
6. The system shows customer history, risk, loyalty, and competitive pressure.
7. A scripted AI narrator produces commercial-style explanations.

---

## Business value

This demo shows how an organization can:

- improve pricing consistency
- reduce manual discounting
- protect margin
- identify customer risk earlier
- support sales teams with commercial insights
- build trust through explainable pricing recommendations
- create a shared language between analytics, sales, and leadership

In short: it turns pricing from a guess into a structured decision process.

---

## Features in detail

### 1) Pricing recommendation engine
Generates a suggested price using:
- customer profile
- product context
- historical purchasing behavior
- regional signals
- competitive pressure
- loyalty / churn indicators

### 2) Customer intelligence view
Shows:
- customer status
- purchase history
- loyalty indicators
- risk profile
- commercial notes
- market position

### 3) Scenario simulation
Lets the user test:
- price increase / decrease
- discount sensitivity
- margin impact
- expected commercial response

### 4) AI-style commercial narration
Provides scripted responses such as:
- why the recommendation is reasonable
- how to position the offer commercially
- how to respond to objections
- what the competitive risk is
- what action should follow next

### 5) Executive dashboard
Includes:
- pricing KPIs
- customer metrics
- pricing traceability
- audit history
- model health indicators
- region-level distribution signals

---

## Sample demo behavior

The application is designed to feel realistic:

- high-loyalty customers tend to receive smaller discounts
- higher competitive pressure leads to more tactical pricing
- high churn risk shifts the recommendation toward retention
- high-ticket products prioritize margin protection
- stronger commercial relationships support value-based pricing

---

## Project structure

```text
pricing-optimization-engine/
├─ app/
│  ├─ api/
│  ├─ core/
│  ├─ pricing_engine/
│  └─ ...
├─ data/
├─ artifacts/
├─ public/
│  ├─ index.html
│  ├─ app.js
│  └─ styles.css
├─ requirements.txt
└─ README.md

# Render Backend: https://pricing-optimization-engine.onrender.com
# Frontend = https://pricing-optimization-engine.pages.dev/




