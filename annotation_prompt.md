1. Process 8 randomly selected images in the datasets/ directory
2. Extract words (without any additional punctuation) in semantically meaningful chunks (like "John Smith" as one chunk instead of two separate words)
3. Create a CSV file for each image with the format:
- Header: ["words", "prediction", "annotation"]
- Words in the "words" column, one per row
- Empty "prediction" and "annotation" columns
4. Name each CSV file after the corresponding image
5. Store all CSV files in the annotations/ directory