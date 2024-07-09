# Destiantion for all of the deps
set(CMAKE_INSTALLATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara_jbrhm/lib)
set(CUPYBARA_LIB_DIR ${CMAKE_BINARY_DIR}/libcupybara.so)

# Install cupybara shared library
install(FILES ${CUPYBARA_LIB_DIR} DESTINATION ${CMAKE_INSTALLATION_DIR})

# Install Shared Library Dependencies
install(CODE [[
        set(CUPYBARA_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara_jbrhm/lib/libcupybara.so)
        set(CMAKE_INSTALLATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara_jbrhm/lib)

        message("Installing: ${CUPYBARA_LIB_DIR}")

        file(GET_RUNTIME_DEPENDENCIES
            LIBRARIES ${CUPYBARA_LIB_DIR}
            RESOLVED_DEPENDENCIES_VAR _r_deps
        )

        message("Installing: " ${_r_deps})

        foreach(_file ${_r_deps})
            file(INSTALL
                DESTINATION "${CMAKE_INSTALLATION_DIR}"
                TYPE SHARED_LIBRARY
                FOLLOW_SYMLINK_CHAIN
                FILES "${_file}"
            )
        endforeach()
      ]])