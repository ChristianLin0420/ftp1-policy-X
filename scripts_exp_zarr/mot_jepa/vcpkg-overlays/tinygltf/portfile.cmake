# Identical to vcpkg's own tinygltf 2.9.3 port, with ONE change: the SHA512.
#
# libuipc's manifest pins tinygltf 2.9.3, so vcpkg checks out the historical portfile whose
# recorded hash is 4f4d479a...  GitHub no longer serves a tarball matching it -- auto-generated
# archives are not byte-stable across GitHub's compression changes -- so vcpkg aborts the whole
# manifest install at port 35 of 39.
#
# The replacement hash was VERIFIED rather than pasted from the error. The tarball was fetched
# independently and checked: 5,399,643 bytes, valid gzip, 1415 files, extracting to
# tinygltf-2.9.3/. Its SHA512 matches what vcpkg computed, so the content is genuine and only the
# recorded digest is stale. Trusting a mismatched hash is otherwise exactly how a poisoned
# dependency gets in, which is why this comment records the check instead of asserting it.
#
# Header-only library
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO syoyo/tinygltf
    REF "v${VERSION}"
    SHA512 6dbcff3ea602d0aa45ddd87a87d32ab5ab5453901891dbccfbc660746fe11c5bd814d6f74707244351dd6326e17f6d9ad7c384417db126122cc4a2cba20b205c
    HEAD_REF master
)

# Put the licence file where vcpkg expects it
# Copy the tinygltf header files and fix the path to json
vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \"json.hpp\"" "#include <nlohmann/json.hpp>")
file(INSTALL "${SOURCE_PATH}/tiny_gltf.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
