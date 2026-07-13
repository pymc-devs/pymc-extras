#   Copyright 2025 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import re

from publish_release_notes_to_discourse import format_release_content


class TestFormatReleaseContent:
    def test_pr_links_are_formatted(self):
        """Test that PR links are formatted correctly in the release body."""
        release_body = """<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### New Features 🎉
* Add time-varying transition matrices to DiscreteMarkovChain by @ricardoV94 in https://github.com/pymc-devs/pymc-extras/pull/700
* Build autoguide in unconstrained space by @theorashid in https://github.com/pymc-devs/pymc-extras/pull/701
### Bugfixes 🪲
* Fix conditional for marginalized DiscreteMarkovChain by @ricardoV94 in https://github.com/pymc-devs/pymc-extras/pull/702

## New Contributors
* @theorashid made their first contribution in https://github.com/pymc-devs/pymc-extras/pull/701

**Full Changelog**: https://github.com/pymc-devs/pymc-extras/compare/v0.3.0...v0.4.0"""

        config = {
            "RELEASE_TAG": "v0.4.0",
            "REPO_NAME": "pymc-devs/pymc-extras",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc-extras/releases/tag/v0.4.0",
        }

        title, content = format_release_content(config)

        assert title == "🚀 Release pymc-extras v0.4.0"

        assert "[#700](https://github.com/pymc-devs/pymc-extras/pull/700)" in content
        assert "[#701](https://github.com/pymc-devs/pymc-extras/pull/701)" in content
        assert "[#702](https://github.com/pymc-devs/pymc-extras/pull/702)" in content

        # No raw PR links should remain outside markdown link syntax
        raw_pr_links = re.findall(
            r"(?<!\()\bhttps://github\.com/pymc-devs/pymc-extras/pull/\d+(?!\))", content
        )
        assert len(raw_pr_links) == 0, f"Found raw PR links: {raw_pr_links}"

        # Other links remain unchanged (e.g., the Full Changelog link)
        assert (
            "**Full Changelog**: https://github.com/pymc-devs/pymc-extras/compare/v0.3.0...v0.4.0"
            in content
        )

    def test_non_repo_links_unchanged(self):
        """Test that PR links from other repositories are not affected."""
        release_body = """Some changes:
* Feature from external repo in https://github.com/other-org/other-repo/pull/123
* Our feature in https://github.com/pymc-devs/pymc-extras/pull/456
"""

        config = {
            "RELEASE_TAG": "v1.0.0",
            "REPO_NAME": "pymc-devs/pymc-extras",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc-extras/releases/tag/v1.0.0",
        }

        _title, content = format_release_content(config)

        assert "[#456](https://github.com/pymc-devs/pymc-extras/pull/456)" in content

        assert "https://github.com/other-org/other-repo/pull/123" in content
        assert "[#123](https://github.com/other-org/other-repo/pull/123)" not in content

    def test_release_structure(self):
        """Test that the overall release structure is correct."""
        config = {
            "RELEASE_TAG": "v1.2.3",
            "REPO_NAME": "pymc-devs/pymc-extras",
            "RELEASE_BODY": "Test body with PR https://github.com/pymc-devs/pymc-extras/pull/999",
            "RELEASE_URL": "https://github.com/pymc-devs/pymc-extras/releases/tag/v1.2.3",
        }

        title, content = format_release_content(config)

        assert title == "🚀 Release pymc-extras v1.2.3"
        assert "A new release of **pymc-extras** is now available!" in content
        assert "**Version:** `v1.2.3`" in content
        assert (
            "**Repository:** [pymc-devs/pymc-extras](https://github.com/pymc-devs/pymc-extras)"
            in content
        )
        assert (
            "**Release Page:** https://github.com/pymc-devs/pymc-extras/releases/tag/v1.2.3"
            in content
        )
        assert "[#999](https://github.com/pymc-devs/pymc-extras/pull/999)" in content

    def test_already_formatted_links_not_double_formatted(self):
        """Test that already-formatted PR links are not double-formatted."""
        release_body = """Some changes:
* Already formatted: [#123](https://github.com/pymc-devs/pymc-extras/pull/123)
* Raw link: https://github.com/pymc-devs/pymc-extras/pull/456
"""

        config = {
            "RELEASE_TAG": "v1.0.0",
            "REPO_NAME": "pymc-devs/pymc-extras",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc-extras/releases/tag/v1.0.0",
        }

        _title, content = format_release_content(config)

        assert "[#123](https://github.com/pymc-devs/pymc-extras/pull/123)" in content
        assert "[#123]([#123](https://github.com/pymc-devs/pymc-extras/pull/123))" not in content

        assert "[#456](https://github.com/pymc-devs/pymc-extras/pull/456)" in content

    def test_user_mentions_converted_to_links(self):
        """Test that user mentions are converted to GitHub profile links."""
        release_body = """## New Contributors
* @ricardoV94 in https://github.com/pymc-devs/pymc-extras/pull/700
* @user-with-dashes contributed in https://github.com/pymc-devs/pymc-extras/pull/701
* @another_user123 helped out in https://github.com/pymc-devs/pymc-extras/pull/702
"""

        config = {
            "RELEASE_TAG": "v0.4.0",
            "REPO_NAME": "pymc-devs/pymc-extras",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc-extras/releases/tag/v0.4.0",
        }

        _title, content = format_release_content(config)

        assert "[@ricardoV94](https://github.com/ricardoV94)" in content
        assert "[@user-with-dashes](https://github.com/user-with-dashes)" in content
        assert "[@another_user123](https://github.com/another_user123)" in content

        assert "[#700](https://github.com/pymc-devs/pymc-extras/pull/700)" in content
