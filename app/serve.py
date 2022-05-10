import requests

prediction = requests.post(
    "http://127.0.0.1:8000/predict",
    headers={"content-type": "application/json"},
    data='{"creation_date":"2015-05-22","component_name":"engine","short_description": "LogTraceException in ProposalUtils.toMethodNam","long_description": "The following incident was reported via the au","assignee_name":"serg.boyko2011","reporter_name":"error-reports-inbox","resolution_category": "fixed","resolution_code":"1","status_category":"closed","status_code":"4","update_date":"2015-05-27","quantity_of_votes":"0", "quantity_of_comments":"0", "resolution_date":"2015-05-27", "bug_fix_time":"2", "severity_category":"normal", "severity_code":"2"}',
).text

print(prediction)