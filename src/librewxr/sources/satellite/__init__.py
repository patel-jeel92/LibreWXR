# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Namespace package for satellite imagery sources.

Each subpackage exposes a ``satellite_provider(settings, cache_dir)``
function returning one or more ``SatelliteContribution`` objects (one
per channel).  The discovery walker in ``librewxr.sources.__init__``
picks them up automatically; see ``collect_satellite_contributions``.
"""
