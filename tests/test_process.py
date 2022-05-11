from src.process import data_selection, component_name_fix, product_name_fix, assignee_fix
import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps


@test_steps(
    "data_selection_step",
    "component_name_fix_step",
    "product_name_fix_step",
    "assignee_fix",
    "final_selection_step"
)


def test_process_suite(test_step, steps_data):
    if test_step == "data_selection_step":
        data_selection_step(steps_data)
    elif test_step == "component_name_fix_step":
        component_name_fix_step(steps_data)
    elif test_step == "product_name_fix_step":
        product_name_fix_step(steps_data)
    elif test_step == "assignee_fix_step":
        assignee_fix_step(steps_data)
    elif test_step == "final_selection_step":
        final_selection_step(steps_data)

def data_selection_step(steps_data):

    data = pd.DataFrame({
            "creation_date": ['2015-05-22', '2018-01-20', '2013-11-25', '2013-11-25'],
            "component_name": ["engine", "UI", "UI", "UI"],
            "product_name": ["product_123", "product_69", "product_69", "product_69"],
            "short_description":["LogTraceException in ProposalUtils.toMethodNam", "[refactoring] Lose comments", "Share Project Wizard dialog typo", "Share Project Wizard dialog typo"],
            "long_description": ["The following incident was reported via the au",  "[refactoring] Lose comments", "Share Project Wizard dialog typo", "Share Project Wizard dialog typo"],
            "assignee_name": ["serg.boyko2011", "sergg.boyko2011", "jdt-ui-inbox", "jdt-ui-inbox"],
            "reporter_name":["error-reports-inbox", "nagrawal", "mickey","mickey"],
            "resolution_category" : ["fixed", "fixed", "fixed","fixed"],
            "resolution_code":[1, 1, 1, 1],
            "status_category": ["closed", "closed", "resolved", "closed"],
            "status_code": [4, 6, 4, 4],
            "update_date":["2015-05-27", "2014-07-23", "2006-12-03", "2006-12-03"],
            "quantity_of_votes":[0, 0, 0, 0],
            "quantity_of_comments":[0, 4, 3, 3],
            "resolution_date": ["2015-05-27", "2018-01-22", "2006-12-03", "2006-12-03"],
            "bug_fix_time": [2, 10, 777, 20],
            "severity_category":["normal", "trivial", "major", "major"],
            "severity_code":[2, 2, 1, 1]
        })
    
    selected_data = data_selection.run(data)
    assert list(selected_data.columns) == [
            "creation_date",
            "component_name",
            "product_name",
            "resolution_category",
            "resolution_code",
            "status_code",
            "update_date",
            "quantity_of_votes",
            "quantity_of_comments",
            "bug_fix_time",
            "severity_code",
            "assignee_name"
    ]
    steps_data.intermediate_a = selected_data

def component_name_fix_step(steps_data):

    intermediate_b = component_name_fix.run(steps_data.intermediate_a)
    steps_data.intermediate_b =intermediate_b
    assert list(intermediate_b['component_name']) == ['other', 'ui', 'ui', 'ui']
    

def product_name_fix_step(steps_data):

    intermediate_c = product_name_fix.run(steps_data.intermediate_b)
    steps_data.intermediate_c = intermediate_c
    assert list(intermediate_c['product_name']) == ['product_other', 'product_69', 'product_69', 'product_69']


def assignee_fix_step(steps_data):

    intermediate_d = assignee_fix.run(steps_data.intermediate_c)
    steps_data.intermediate_d = intermediate_d
    assert list(intermediate_d['assignee_name']) == ["serg.boyko2011", "serg.boyko2011", "jdt-ui-inbox", "jdt-ui-inbox"]

def final_selection_step(steps_data):

    schema = DataFrameSchema(
        {
            "component_name": Column(object),
            "product_name": Column(object),
            "resolution_code": Column(int, Check.greater_than_or_equal_to(0)),
            "status_code": Column(int, Check.greater_than_or_equal_to(0)),
            "quantity_of_votes": Column(int, Check.greater_than_or_equal_to(0)),
            "quantity_of_comments": Column(int, Check.greater_than_or_equal_to(0)),
            "bug_fix_time": Column(int, Check.greater_than_or_equal_to(0)),
            "severity_code": Column(int, Check.greater_than_or_equal_to(0)),
            "assignee_name": Column(object)
        }
    )
    schema.validate(steps_data.intermediate_c)


    


















