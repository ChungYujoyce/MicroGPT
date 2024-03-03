import argparse
import os
import sys
import tempfile

import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)
app.secret_key = "LeafmanZSecretKey"

API_HOST = "http://localhost:5110/api"


# PAGES #
@app.route("/", methods=["GET", "POST"])
def home_page():
    if request.method == "POST":
        global original_result
        if "user_prompt" in request.form:
            user_prompt = request.form.get("user_prompt")
            print(f"User Prompt: {user_prompt}")

            main_prompt_url = f"{API_HOST}/prompt_route"
            response = requests.post(main_prompt_url, data={"user_prompt": user_prompt})
            original_result = response
            print(response)
            if response.status_code == 200:
                # print(response.json())  # Print the JSON data from the response
                return render_template("home.html", show_response_modal=True, response_dict=response.json())
        elif "documents" in request.files:
            if request.form.get("action") == "reset":
                delete_source_url = f"{API_HOST}/delete_source"  # URL of the /api/delete_source endpoint
                response = requests.get(delete_source_url)

            save_document_url = f"{API_HOST}/save_document"
            files = request.files.getlist("documents")
            for file in files:
                print(file.filename)
                filename = secure_filename(file.filename)
                with tempfile.SpooledTemporaryFile() as f:
                    f.write(file.read())
                    f.seek(0)
                    response = requests.post(save_document_url, files={"document": (filename, f)})
                    print(response)  # print HTTP response status code for debugging
                    
            if request.form.get("action") == "add":
                run_ingest_url = f"{API_HOST}/run_add"

            if request.form.get("action") == "reset":
                run_ingest_url = f"{API_HOST}/run_reset"  # URL of the /api/run_ingest endpoint
            # Make a GET request to the /api/run_ingest endpoint
            response = requests.post(run_ingest_url)
            print(response)

        elif "editInput" in request.form:
            id, revise_result = request.form.get("id"), request.form.get("revise_result")
            print(f"id: {id}")
            print(f"Revised content: {revise_result}")
            revise_result_url = f"{API_HOST}/run_update"
            response = requests.put(revise_result_url, data={"id": id, "revise_result": revise_result})
            print(response.content)
            
        elif "deleteInput" in request.form:
            id = request.form.get("id"),
            print(f"id: {id}")
            revise_result_url = f"{API_HOST}/run_delete"
            response = requests.delete(revise_result_url, data={"id": id})
            print(response.content)
            

    # Display the form for GET request
    return render_template(
        "home.html",
        show_response_modal=False,
        response_dict={"Prompt": "None", "Answer": "None", "Sources": [("ewf", "wef")]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3111, help="Port to run the UI on. Defaults to 5111.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the UI on. Defaults to 127.0.0.1. "
        "Set to 0.0.0.0 to make the UI externally "
        "accessible from other devices.",
    )
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
