# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Constants for ngff-zarr package."""

# Supported NGFF specification versions
SUPPORTED_VERSIONS = ["0.4", "0.5", "0.6"]

# Version mapping for conversion compatibility
VERSION_ALIASES = {
    "0.4": "0.4",
    "0.5": "0.5", 
    "0.6": "0.6",
    "latest": "0.6"
}