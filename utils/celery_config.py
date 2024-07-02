from os import getenv

result_backend = getenv("REDIS_URL", "redis://127.0.0.1:6379")
broker_url = getenv("REDIS_URL", "redis://127.0.0.1") + "/0"

task_serializer = "json"
result_serializer = "json"
timezone = "Asia/Taipei"

task_time_limit=60
task_soft_time_limit=60