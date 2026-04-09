from app.api.main import app

@app.get("/debug/routes")
def debug_routes():
    return [route.path for route in app.routes]
