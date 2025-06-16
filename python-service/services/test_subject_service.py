from db import db

# Reference the test_subjects collection
if db is not None:
    test_subjects_collection = db["testsubjects"]

def find_test_subject_by_id(subject_id):
    """Find a test subject by its ID."""
    return test_subjects_collection.find_one({"subjectId": subject_id})

def create_test_subject(subject_id, data):
    """Create a new test subject."""
    return test_subjects_collection.insert_one({"subjectId": subject_id, "data": data})

def update_test_subject(subject_id, data):
    """Update an existing test subject."""
    return test_subjects_collection.update_one(
        {"subjectId": subject_id},
        {"$set": {"data": data}}
    )