from flask import Blueprint, request, jsonify
from services.test_subject_service import (
    find_test_subject_by_id,
    create_test_subject,
    update_test_subject,
)

test_subject_blueprint = Blueprint("test_subject", __name__)

@test_subject_blueprint.route("/python-service/upload-test-subject", methods=["POST"])
def upload_test_subject():
    """API endpoint to upload or update a test subject."""
    data = request.json
    subject_id = data.get("subjectId")
    subject_data = data.get("data")

    if not subject_id or not subject_data:
        return jsonify({"error": "Missing required fields: subjectId or data"}), 400

    try:
        # Check if the subject already exists
        existing_subject = find_test_subject_by_id(subject_id)

        if existing_subject:
            # Update the existing subject's data
            updated_data = {**existing_subject["data"], **subject_data}  # Merge existing data with new data
            update_test_subject(subject_id, updated_data)
            return jsonify({"message": "Subject updated successfully"}), 200
        else:
            # Create a new subject
            create_test_subject(subject_id, subject_data)
            return jsonify({"message": "Subject created successfully"}), 201
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Error uploading subject data"}), 500