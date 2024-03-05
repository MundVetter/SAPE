def read_and_group_lines(file_path):
    """
    Reads lines from the given file and groups them by the category.
    """
    lines_by_category = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            category = line.split(':')[-1]
            if category.isdigit() and 1 <= int(category) <= 20:
                if category not in lines_by_category:
                    lines_by_category[category] = []
                lines_by_category[category].append(line)
    return lines_by_category

def distribute_to_files(lines_by_category):
    """
    Distributes lines into separate files ensuring no category is repeated across files.
    Each file can contain multiple categories, but each category is unique across all files.
    """
    files = []  # This will store the list of categories for each file

    # Attempt to place each category into a file
    for category, lines in lines_by_category.items():
        added_to_file = False
        for file_categories in files:
            if category not in file_categories:
                file_categories.add(category)
                with open(f'unique_categories_{files.index(file_categories) + 1}.ts', 'a') as f:
                    f.write('\n'.join(lines) + '\n')
                added_to_file = True
                break
        if not added_to_file:
            # Create a new file for this category
            new_file_index = len(files) + 1
            with open(f'unique_categories_{new_file_index}.ts', 'w') as f:
                f.write('\n'.join(lines) + '\n')
            files.append({category})

input_file_path = 'CharacterTrajectories_TEST.ts'  # Update this with the actual file path
lines_by_category = read_and_group_lines(input_file_path)
distribute_to_files(lines_by_category)
