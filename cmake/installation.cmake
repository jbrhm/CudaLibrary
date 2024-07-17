# Destiantion for all of the deps
set(CUPYBARA_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara)
set(CUPYBARA_INSTALLATION_DIR ${CUPYBARA_PACKAGE_DIR}/lib)
set(CUPYBARA_LIB_DIR ${CMAKE_BINARY_DIR}/libcupybara.so)
file(
        RELATIVE_PATH CUPYBARA_INSTALLATION_DIR_STR
        ${CUPYBARA_PACKAGE_DIR}
        "${CUPYBARA_INSTALLATION_DIR}/libcupybara.so"  
    )

# Configure the path so cupybara knows where to find the libraries
configure_file(cmake/cupybara_paths.py.in cupybara_paths.py @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cupybara_paths.py DESTINATION ${CUPYBARA_PACKAGE_DIR})

# Install python source
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/cupybara.py DESTINATION ${CUPYBARA_PACKAGE_DIR})

# Install cupybara shared library
install(FILES ${CUPYBARA_LIB_DIR} DESTINATION ${CUPYBARA_INSTALLATION_DIR})


# Install Shared Library Dependencies
install(CODE [[
        set(CUPYBARA_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara/lib/libcupybara.so)
        set(CUPYBARA_INSTALLATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package/cupybara/lib)

        message("Installing: ${CUPYBARA_LIB_DIR}")

        file(GET_RUNTIME_DEPENDENCIES
            LIBRARIES ${CUPYBARA_LIB_DIR}
            RESOLVED_DEPENDENCIES_VAR _r_deps
        )

        message("Installing: " ${_r_deps})

        foreach(_file ${_r_deps})
            file(INSTALL
                DESTINATION "${CUPYBARA_INSTALLATION_DIR}"
                TYPE SHARED_LIBRARY
                FOLLOW_SYMLINK_CHAIN
                FILES "${_file}"
            )
        endforeach()
      ]])

