# Contributing to GeoPINN Studio

Thank you for your interest in contributing to GeoPINN Studio.

## How to Contribute

### Bug Reports
Open an issue with:
- OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- Log output (from the console panel)

### Feature Requests
Open an issue tagged `enhancement` with:
- Use case description
- Expected workflow
- Geophysical method context (if applicable)

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Follow existing code style (Python: PEP8, JS: ESLint config)
4. Add tests if applicable
5. Update `CHANGELOG.md`
6. Submit PR against `main`

## Development Setup

```bash
# Backend
cd geopinn-backend
pip install -r requirements.txt
uvicorn server:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd geopinn-frontend
npm install
npm run dev
```

## Code Structure

```
geopinn-backend/
  engines/          # Physics forward engines
  server.py         # FastAPI endpoints
geopinn-frontend/
  src/App.jsx       # Main React application
  main.cjs          # Electron main process
```

## Contact

geopinnstudio@geopinn.tr
