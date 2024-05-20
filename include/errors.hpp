#pragma once

#include <exception>
#include <string>

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
