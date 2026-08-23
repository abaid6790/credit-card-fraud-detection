import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    if not getattr(app, "model_ready", False):
        print("=" * 70)
        print("WARNING: No trained model found.")
        print("The app will still start, but predictions will return 503 until you run:")
        print("    python ml/train.py")
        print("(after placing data/creditcard.csv — see data/README.md)")
        print("=" * 70)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") != "production")
