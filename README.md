# National-Grid
Mini Project

A model that would forecast next 12 months of Emergency and Normal service calls.
Data: The dataset provided is related to service calls, specifically when they were received, the priority, and the town where the service is needed. 
The priority is contained within the request_type column and is either an emergency (E) or normal (N). This data is spread across two tables:
              1. RECEIVED: Date and time when the call was received
              2. REQ_INFO: Request type and town for the call
The data is provided as a SQLite DB.
