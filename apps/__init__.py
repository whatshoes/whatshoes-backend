from flask import Flask

def create_app():
    app = Flask(__name__)

    from .routes import routes_list
    routes_list(app)
    return app
