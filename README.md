# Customer Churn Prediction Web App

This Flask app serves the trained Assignment 1 churn model. The saved model is a scikit-learn pipeline: it applies the same imputing, scaling, and one-hot encoding used during training before predicting. The web form collects the model's customer features and displays a human-readable churn estimate.

The submission bundle is [Assignment_2_Project.zip](./Assignment_2_Project.zip). The completed local work and browser examples are documented in [Assignment_2_Final_Report.pdf](./Assignment_2_Final_Report.pdf). PythonAnywhere deployment is still pending account setup and a public URL.

## Run locally

1. Use a compatible Python environment and install packages with `pip install -r requirements.txt`.
2. Run the notebook through its model-export cell to create `model.joblib` in this project folder.
4. Start the app with `python app.py`.
5. Open `http://127.0.0.1:5000`.

The app fails clearly at startup if `model.joblib` is missing or does not match the form's feature order. Keep the model artifact and the pinned scikit-learn version together.

## Test

Run the Flask route and input-validation tests with:

```powershell
python -m unittest discover -s tests -v
```

## Deploy to PythonAnywhere

1. Upload the project, including `app.py`, `model.joblib`, `requirements.txt`, `templates/`, and `static/`.
2. Create a PythonAnywhere web app using a Python version compatible with the pinned dependencies.
3. Create/activate a virtual environment and install `requirements.txt`.
4. Set the WSGI file to import `app` from this project folder, for example:

   ```python
   import sys
   project_home = "/home/YOUR_USERNAME/Churn-Prediction"
   if project_home not in sys.path:
       sys.path.insert(0, project_home)
   from app import app as application
   ```

5. Reload the web app and test the public URL with valid, missing, and out-of-range inputs.

The account username and deployed URL are specific to the owner and must be filled in after deployment.
