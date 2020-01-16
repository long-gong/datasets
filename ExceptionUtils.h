#ifndef _DATASET_EXCEPTION_H_
#define _DATASET_EXCEPTION_H_
#include <stdexcept>

class DSException : public std::runtime_error {
 public:
  DSException(const std::string &rFileName,  // filename
                unsigned int nLineNumber,      // line number
                const std::string &rMessage    // error message
                )
      : std::runtime_error(rFileName + ":" + std::to_string(nLineNumber) +
                           ": " + rMessage) {}
};

#define DS_REQUIRED(C)                                                   \
  do {                                                                     \
    if (!(C)) throw DSException(__FILE__, __LINE__, #C " is required!"); \
  } while (false)

#define DS_REQUIRED_MSG(C, M)                                        \
  do {                                                                 \
    if (!(C))                                                          \
      throw DSException(                                             \
          __FILE__, __LINE__,                                          \
          std::string(#C " is required! Message: ") + std::string(M)); \
  } while (false)

#endif  // _DATASET_EXCEPTION_H_