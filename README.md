# Alimento

AI-powered nutrition platform with secure sign-in, meal intelligence, planning, and progress tracking.

Alimento helps people make better food decisions by turning meal data into practical guidance. Users can analyze meals, track nutrition trends, build plans, and receive personalized coaching in one web application.

---

## About This Fork

This project was originally developed as a team application.
I forked and extended it to improve functionality, reliability, and overall user experience.

Focused on improving real-time data handling, system reliability, and user workflow efficiency, this fork introduces multiple enhancements across UI, backend reliability, and data processing.

---

## Key Improvements

* **Dashboard Enhancements**
  Introduced month-based calendar navigation with day-level controls (previous / next / today) for seamless tracking of meals and goals.

* **Smart Inventory System**
  Added categorized storage (pantry / fridge / freezer), low-stock and expiration tracking, bulk actions, and AI-powered meal suggestions based on available ingredients.

* **Recipe Management**
  Improved usability with profile-aware context, streamlined diet workflows, and bulk deletion for user-owned recipes.

* **System Reliability & Performance**

  * Configured MongoDB Atlas connections using `certifi` CA bundle for consistent TLS handling
  * Implemented retry + backoff strategies for Gemini API calls
  * Optimized barcode lookup with caching and normalized queries

---

## Active Development

Ongoing work is organized across feature branches:

* `feature/pratham-improvements` – UI, dashboard, and workflow enhancements
* *(additional branches will be added as development continues)*

To explore a branch locally:

```bash
git clone https://github.com/rathodpratham15/alimento.git
cd alimento
git checkout <branch-name>
```

---

## Product Overview

Alimento is a full-stack web application focused on day-to-day nutrition decision support.

* Analyze meals from image URL, text, barcode, or manual nutrition input
* Convert meal logs into calorie and macro insights
* Build weekly plans and reusable recipes
* Track weight and consistency over time
* Manage reminder settings and integration readiness

---

## Core Capabilities

* **AI-Powered Analysis**: Instant nutrition breakdown (Calories, Macros, Micros) using Gemini 2.5 Flash Lite
* **Global Dashboard**: Timezone-aware tracking for meals & hydration
* **Personalized Goals**: Smart scoring & insights for Keto, Vegan, Paleo, and more
* **Secure & Private**: Google OAuth, private history, and guest sessions
* **Modern UX**: Responsive Glassmorphism design with Dark Mode

---

## Platform Features

* **Unified Meal Logging**: Image URL, text input, barcode, and manual macro logging
* **Weekly Meal Planner**: Editable plans with auto grocery list generation
* **Recipe Library**: Private/public recipes with nutrition data
* **Progress Tracking**: Daily trends and body-weight logging
* **Social Challenges**: Leaderboard-based challenges
* **Settings & Integrations**: Persisted user preferences
* **Migration Support**: Idempotent migration from `analysis_history` to `meal_logs`

---

![Alimento Interface](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-green)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen)

---

## Quick Setup

### Prerequisites

* Python 3.12+
* Google AI API key (Gemini)
* MongoDB Atlas account
* Google Cloud Console project (OAuth)

### Installation

```bash
git clone https://github.com/rathodpratham15/alimento.git
cd alimento
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_ai_api_key
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/alimento

GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
OAUTH_REDIRECT_URI=http://localhost:5001/auth/callback
FLASK_SECRET_KEY=your_secure_secret_key

ADMIN_EMAILS=admin@example.com
```

### Run Application

```bash
python app.py
```

Access at: http://localhost:5001

---

## Tech Stack

### Backend

* Flask 2.3.2
* Google Gemini 2.5 Flash Lite
* MongoDB Atlas + PyMongo
* Flask-Login, Authlib, Flask-WTF, Flask-CORS
* Pillow

### Frontend

* Tailwind CSS
* Phosphor Icons
* Geist + Playfair Display
* Vanilla JavaScript
* Responsive Design

### Security & Auth

* Google OAuth 2.0
* Session cookies
* CSRF protection
* Rate limiting

### Deployment

* Vercel-ready
* Environment-based configuration
* Production security settings

---

## Security Features

### Authentication

* Google OAuth 2.0
* Secure session management
* HTTPS-ready
* CSRF protection

### Privacy

* User data isolation
* Guest session isolation
* Secure tracking
* No cross-user leakage

### Rate Limiting

* Daily usage limits
* Per-user tracking
* Graceful limit handling

---

## Deployment

### Vercel (Recommended)

1. Connect repository
2. Add environment variables
3. Deploy on push

### Manual Deployment

1. Set `PRODUCTION=true`
2. Configure HTTPS
3. Update OAuth redirects
4. Monitor performance

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request