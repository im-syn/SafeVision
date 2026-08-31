"""WSGI entry point used by Waitress, Gunicorn, and other production servers."""

from app import app

__all__ = ["app"]

