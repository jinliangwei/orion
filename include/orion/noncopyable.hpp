#pragma once

#define DISALLOW_COPY(CLASS_NAME) \
  CLASS_NAME(const CLASS_NAME &other) = delete; \
  CLASS_NAME& operator = (const CLASS_NAME &) = delete
