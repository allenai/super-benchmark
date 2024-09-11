from typing import Dict, Tuple

from super.env import Environment
from super.env.aci import ACI


def find_matches(content_lines, context_lines_before_patch, replaced_lines, strip: bool = False):
    compare_with = context_lines_before_patch + replaced_lines
    old_len = len(compare_with)
    indices = []
    partial_or_full_match = None
    best_line_matches_so_far = []
    for i in range(len(content_lines) - old_len + 1):
        line_matches = []
        for j in range(old_len):
            content_line = content_lines[i + j]
            compare_with_line = compare_with[j]

            if strip:
                content_line = content_line.strip()
                compare_with_line = compare_with_line.strip()

            if content_line == compare_with_line or (
                    len(content_line.strip()) == 0 and len(compare_with_line.strip()) == 0):
                line_matches.append(content_line)

        if len(line_matches) == old_len:
            indices.append(i + len(context_lines_before_patch))
            partial_or_full_match = content_lines[i:i + old_len]
        elif len(line_matches) > 0:
            # We keep the longest partial matches in case there are no full matches
            if partial_or_full_match and len(line_matches) <= len(best_line_matches_so_far):
                # not longer than the previous match
                continue
            if not any(len(line.strip()) > 0 for line in line_matches):
                # only whitespace matches
                continue
            best_line_matches_so_far = line_matches
            partial_or_full_match = content_lines[i:i + old_len]
    return indices, partial_or_full_match


class EditACI(ACI):
    """
    A wrapper/ACI (agent-computer interface) for editing files.
    """
    def __init__(self, env: Environment, detailed_editing_feedback=False):
        super().__init__(env)
        self._detailed_editing_feedback = detailed_editing_feedback

    def step(self, action: Dict[str, str]) -> Tuple[str, bool]:
        if action["type"] == "edit":
            values = action["content"].split("\n")
            if len(values) < 3:
                return "Invalid format. Correct format: {filename}\n{optional preceding line}\n<<<BEFORE_EDIT>>>\n{lines to remove}\n<<<AFTER_EDIT>>>\n{lines to add}", False

            filename, content = values[0], "\n".join(values[1:])
            return self._edit_file(filename, content, allow_whitespace_diffs=False), False
        else:
            raise ValueError(f"Invalid action type: {action['type']}")

    def _edit_file(self, file: str, content: str, allow_whitespace_diffs: bool = False) -> str:
        lines = content.rstrip("\n").splitlines()

        error_message = """Format of edit is:```
{filename}
[optional] preceding lines
<<<BEFORE_EDIT>>>
line to remove
line to remove
<<<AFTER_EDIT>>>
line to add
line to add
line to add
```"""

        # Parse the patch to identify the line to remove and the line to add
        context_lines_before_patch = []
        replaced_lines = []
        new_lines = []
        passed_start_cursor = False
        passed_end_cursor = False
        for i, line in enumerate(lines):
            if line.strip() == "<<<BEFORE_EDIT>>>":
                if passed_start_cursor:
                    return f"Found multiple BEFORE_EDIT in the patch. Line {i + 1}: {line}\n{error_message}"
                passed_start_cursor = True
            elif line.strip() == "<<<AFTER_EDIT>>>":
                if passed_end_cursor:
                    return f"Found multiple AFTER_EDIT in the patch. Line {i + 1}: {line}\n{error_message}"
                passed_end_cursor = True
            else:
                if passed_start_cursor and not passed_end_cursor:
                    replaced_lines.append(line)
                elif not passed_start_cursor:
                    context_lines_before_patch.append(line)
                else:
                    new_lines.append(line)

        if not replaced_lines and not new_lines:
            return f"Could not find any lines to replace in the patch. {error_message}"

        # Split the original content into lines
        original_file_contents = self._env.run_command(f"!cat {file}")

        if "No such file or directory" in original_file_contents:
            return f"ERROR: File {file} does not exist."

        content_lines = [l.strip("\r") for l in original_file_contents.split("\n")]

        # Locate the block of context + replaced lines in the file content

        indices, matched_lines = find_matches(content_lines, context_lines_before_patch, replaced_lines, strip=False)

        if len(indices) == 0:
            output = f"Could not find the following lines to replace in the file content:\n```\n"
            output += "\n".join(context_lines_before_patch + replaced_lines) + "\n```"

            if self._detailed_editing_feedback:
                indices, matched_lines_stripped = find_matches(content_lines, context_lines_before_patch, replaced_lines, strip=True)

                if allow_whitespace_diffs and len(indices) == 1:
                    # since we allow whitespace diffs, we can go ahead with the edit
                    pass
                else:
                    if indices:
                        output += "\nDid you mean to replace the following lines (notice leading/trailing whitespaces difference)?\n```\n"
                        output += "\n".join(matched_lines_stripped) + "\n```"
                    else:
                        if matched_lines:
                            output += "\nHere are partially matched lines:\n```\n"
                            output += "\n".join(matched_lines) + "\n```"
            return output
        elif len(indices) > 1:
            error = f"Found multiple ({len(indices)}) occurrences of the <<<BEFORE_EDIT>>>  lines. Add 1-3 lines before or after these lines to replace to disambiguate."

            if self._detailed_editing_feedback:
                error += f"\nHere are the first two occurrences with additional context, did you mean one of these?"
                for i in range(2):
                    error += f"\nOccurrence {i + 1}:\n```\n"
                    start_index = indices[i] - 2
                    end_index = indices[i] + len(replaced_lines) + 2
                    for line in range(max(start_index, 0), min(end_index, len(content_lines))):
                        error += f"\n{content_lines[line]}"
                    error += "\n```"
            return error

        # Replace the block in the content
        start_index = indices[0]
        end_index = start_index + len(replaced_lines)
        updated_content = content_lines[:start_index] + new_lines + content_lines[end_index:]

        # Write the updated content back to the file
        updated_content_str = "\n".join(updated_content)
        self._env.run_command(f"%%writefile {file}\n{updated_content_str}")

        if self._detailed_editing_feedback:
            output = "Edit was successful. Here's the relevant content *after edit* for your inspection (you should check indentation is correct):"
            for line in range(max(start_index - 1 - 6, 0), min(start_index + len(new_lines) + 6, len(updated_content))):
                output += f"\n{updated_content[line]}"
        else:
            output = "Edit was successful."

        return output

