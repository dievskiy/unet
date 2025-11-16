#!/usr/bin/env python3
"""Print CloudWatch log group/stream names for a SageMaker endpoint."""
from __future__ import annotations

import argparse

import boto3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("endpoint_name")
args = parser.parse_args()

sm = boto3.client("sagemaker")
endpoint = sm.describe_endpoint(EndpointName=args.endpoint_name)

logs = boto3.client("logs")
log_group = f"/aws/sagemaker/Endpoints/{args.endpoint_name}"
streams = logs.describe_log_streams(logGroupName=log_group, orderBy="LastEventTime", descending=True)
print({"LogGroup": log_group, "Streams": [s["logStreamName"] for s in streams.get("logStreams", [])]})
