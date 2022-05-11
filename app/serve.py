import requests

prediction = requests.post(
    "http://127.0.0.1:8000/predict",
    headers={"content-type": "application/json"},
    data='{"creation_date":"2015-05-22","component_name":"core","short_description": "LogTraceException in ProposalUtils.toMethodNam","long_description": "The following incident was reported via the au","assignee_name":"recommenders-inbox","reporter_name":"error-reports-inbox","resolution_category": "fixed","resolution_code":"1","status_category":"closed","status_code":"6","update_date":"2015-05-27","quantity_of_votes":"0", "quantity_of_comments":"2", "resolution_date":"2015-05-27", "bug_fix_time":"5", "severity_category":"normal", "severity_code":"2"}',
).text

print(prediction)
