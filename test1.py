import spacy
import inflect

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Define the 80 categories from the YOLOv5 model
categories = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Category synonyms dictionary
category_synonyms = {
    "person": {"individual", "someone", "somebody", "mortal", "soul", "human", "human being", "human person", "citizen", "man", "woman", "child", "adult"},
    "bicycle": {"bike", "cycle", "two-wheeler"},
    "car": {"automobile", "vehicle", "auto"},
    "motorcycle": {"motorbike", "bike"},
    # Add more categories with their synonyms as needed
}


# Create a reverse mapping from synonym to main category
synonym_to_category = {}
for category, synonyms in category_synonyms.items():
    for synonym in synonyms:
        synonym_to_category[synonym] = category

# Convert categories to a set for faster lookup
categories_set = set(categories)

# Initialize the inflect engine
p = inflect.engine()

def find_matching_categories(text):
    # Process the text with spacy
    doc = nlp(text.lower())

    include_list = set()
    exclude_list = set()

    negated = False
    for token in doc:
        # Check for negations
        if token.dep_ == "neg":
            negated = True
            continue
        
        # Check for conjunctions that might reset the negation scope
        if token.dep_ in ("cc", "conj"):
            negated = False

        singular_form = p.singular_noun(token.text) if p.singular_noun(token.text) else token.text
        if token.text in synonym_to_category or singular_form in synonym_to_category:
            main_category = synonym_to_category.get(token.text, synonym_to_category.get(singular_form))
            if negated:
                exclude_list.add(main_category)
            else:
                include_list.add(main_category)
        elif token.text in categories_set or singular_form in categories_set:
            main_category = token.text if token.text in categories_set else singular_form
            if negated:
                exclude_list.add(main_category)
            else:
                include_list.add(main_category)

    # Check for multi-word categories
    tokens = [token.text for token in doc]
    for category in categories:
        category_tokens = category.split()
        for i in range(len(tokens) - len(category_tokens) + 1):
            if tokens[i:i+len(category_tokens)] == category_tokens:
                if any(doc[j].dep_ == "neg" for j in range(i)):
                    exclude_list.add(category)
                else:
                    include_list.add(category)

    # Subtract exclude_list from include_list to get the final list
    matching_categories = list(include_list - exclude_list)

    return matching_categories

# Example usage
input_text = input("Enter text: ")
matches = find_matching_categories(input_text)
print("Matching categories:", matches)
