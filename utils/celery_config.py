from os import getenv

#docker ver (not implemented)
# result_backend = "redis://172.23.0.2"
# broker_url = "redis://172.23.0.2:6379/0"

#local ver
result_backend = getenv("REDIS_URL", "redis://127.0.0.1:6379")
broker_url = getenv("REDIS_URL", "redis://127.0.0.1") + "/0"

task_serializer = "json"
result_serializer = "json"
timezone = "Asia/Taipei"