#pragma once

#include <exception>
#include <string>

namespace tiledcuda::errors {

class NotImplementedException : public std::exception {
  public:
    NotImplementedException(const char* error = "Not yet implemented!") {
        errorMessage = error;
    }

    // Provided for compatibility with std::exception.
    const char* what() const noexcept { return errorMessage.c_str(); }

  private:
    std::string errorMessage;
};

}  // namespace tiledcuda::errors
