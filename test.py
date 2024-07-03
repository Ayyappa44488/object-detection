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
    "person": {"individual", "someone", "somebody", "mortal", "soul", "human", "human being", "human person", "citizen", "man", "woman", "child", "adult", "individual"},
    "bicycle": { "cycle", "pushbike", "velocipede", "tandem", "mountain bike", "road bike", "racer", "cyclist's vehicle"},
    "car": {"automobile", "vehicle", "auto", "machine", "motorcar", "sedan", "coupe", "convertible", "hatchback", "SUV", "van"},
    "motorcycle": {"bike", "motorbike", "chopper", "scooter", "moped", "two-wheeler", "dirt bike", "motorcross", "hog"},
    "airplane": {"plane", "aeroplane", "aircraft", "jet", "airliner", "propeller plane", "biplane", "seaplane", "jumbo jet", "fighter plane"},
    "bus": {"coach", "public transport", "shuttle", "transit bus", "city bus", "school bus", "minibus", "double-decker", "tour bus", "charter bus"},
    "train": {"locomotive", "railroad", "railway", "subway", "monorail", "bullet train", "freight train", "commuter train", "metro", "express train"},
    "truck": {"lorry", "hauler", "pickup", "van", "delivery truck", "big rig", "semi", "tanker", "tractor-trailer", "freight truck"},
    "boat": {"ship", "vessel", "watercraft", "yacht", "sailboat", "motorboat", "ferry", "canoe", "kayak", "barge"},
    "traffic light": {"signal", "stoplight", "traffic signal", "traffic control", "red light", "green light", "yellow light", "road signal", "traffic lamp", "crossing light"},
    "fire hydrant": {"hydrant", "fireplug", "standpipe", "water outlet", "water source", "fire water", "emergency water supply", "firefighting outlet", "fire post", "hydrant post"},
    "stop sign": {"halt sign", "stop marker", "traffic sign", "stop board", "stop signal", "road sign", "halt marker", "halt signal", "traffic stop", "red sign"},
    "parking meter": {"meter", "parking meter machine", "pay-and-display", "ticket machine", "parking payment", "parking ticket machine", "pay station", "parking fee machine", "parking toll", "parking payment station"},
    "bench": {"seat", "pew", "park bench", "garden bench", "outdoor seat", "resting place", "public seat", "long seat", "wooden seat", "sitting place"},
    "bird": {"avian", "fowl", "feathered friend", "winged animal", "songbird", "birdie", "raptor", "waterfowl", "game bird", "nestling"},
    "cat": {"feline", "kitty", "kitten", "tomcat", "tabby", "mouser", "house cat", "domestic cat", "stray cat", "puss"},
    "dog": {"canine", "puppy", "hound", "mutt", "pooch", "cur", "man’s best friend", "doggy", "pup", "mongrel"},
    "horse": {"steed", "equine", "stallion", "mare", "colt", "filly", "foal", "pony", "nag", "gelding"},
    "sheep": {"ovine", "lamb", "ewe", "ram", "wether", "yearling", "woolly", "flock", "shepherd's animal", "baa"},
    "cow": {"bovine", "cattle", "heifer", "bull", "steer", "calf", "dairy cow", "ox", "milker", "bovidae"},
    "elephant": {"pachyderm", "jumbo", "tusker", "bull elephant", "cow elephant", "baby elephant", "calf", "Asian elephant", "African elephant", "trunked mammal"},
    "bear": {"ursus", "grizzly", "polar bear", "black bear", "brown bear", "cub", "panda", "koala", "sloth bear", "spectacled bear"},
    "zebra": {"striped horse", "equid", "wild horse", "African horse", "plain zebra", "mountain zebra", "Grevy's zebra", "herd animal", "hooved animal", "savanna animal"},
    "giraffe": {"long-necked animal", "tall mammal", "ungulate", "African mammal", "savanna animal", "herbivore", "towering animal", "spotted mammal", "giraffid", "necked animal"},
    "backpack": {"knapsack", "rucksack", "pack", "haversack", "daypack", "satchel", "bookbag", "schoolbag", "bag", "carryall"},
    "umbrella": {"brolly", "parasol", "sunshade", "rainshade", "shade", "canopy", "rain cover", "shelter", "portable shelter", "weather protection"},
    "handbag": {"purse", "bag", "pocketbook", "tote", "clutch", "shoulder bag", "satchel", "carryall", "hand-held bag", "fashion accessory"},
    "tie": {"necktie", "cravat", "bow tie", "ascot", "neckerchief", "scarf", "bolo tie", "string tie", "fashion accessory", "formal wear"},
    "suitcase": {"luggage", "baggage", "travel bag", "valise", "carry-on", "trunk", "overnight bag", "grip", "travel case", "portable case"},
    "frisbee": {"disc", "flying disc", "flying saucer", "throwing disc", "recreational disc", "plastic disc", "frisbee disc", "play disc", "toy disc", "sport disc"},
    "skis": {"skiing equipment", "snow skis", "cross-country skis", "alpine skis", "downhill skis", "ski poles", "ski gear", "ski set", "ski pair", "winter sports gear"},
    "snowboard": {"snowboarding equipment", "snow board", "winter board", "ski board", "snow gear", "snow sport board", "freestyle board", "alpine board", "board", "riding board"},
    "sports ball": {"ball", "game ball", "play ball", "athletic ball", "soccer ball", "basketball", "baseball", "football", "tennis ball", "volleyball"},
    "kite": {"flying toy", "kite flyer", "kite flying", "stringed toy", "wind toy", "paper kite", "plastic kite", "airborne toy", "sky toy", "wind flyer"},
    "baseball bat": {"bat", "sports bat", "wooden bat", "metal bat", "slugger", "club", "baseball equipment", "hitting bat", "playing bat", "athletic bat"},
    "baseball glove": {"mitt", "glove", "catcher’s mitt", "fielder’s glove", "baseball gear", "baseball equipment", "leather glove", "sporting glove", "playing glove", "athletic glove"},
    "skateboard": {"board", "skating board", "roller board", "freestyle board", "street board", "trick board", "deck", "longboard", "shortboard", "ride board"},
    "surfboard": {"surfing board", "wave board", "riding board", "water board", "beach board", "longboard", "shortboard", "surf gear", "surf equipment", "water sports board"},
    "tennis racket": {"racket", "tennis equipment", "sports racket", "racquet", "playing racket", "tennis gear", "tennis bat", "tennis tool", "athletic racket", "net game racket"},
    "bottle": {"flask", "container", "jar", "vessel", "canister", "carafe", "jug", "decanter", "glass", "plastic bottle"},
    "wine glass": {"goblet", "stemware", "drinking glass", "wine goblet", "wine cup", "chalice", "tumbler", "wine vessel", "glass", "wine container"},
    "cup": {"mug", "drinking cup", "tumbler", "beaker", "chalice", "vessel", "container", "teacup", "coffee cup", "drinking vessel"},
    "fork": {"cutlery", "utensil", "dining tool", "eating implement", "tableware", "silverware", "pronged tool", "pronged utensil", "food tool", "dining fork"},
    "knife": {"blade", "cutlery", "utensil", "dining tool", "eating implement", "tableware", "silverware", "cutting tool", "sharp implement", "cooking knife"},
    "spoon": {"utensil", "cutlery", "eating implement", "dining tool", "tableware", "silverware", "scoop", "dipper", "ladle", "serving spoon"},
    "bowl": {"dish", "container", "basin", "vessel", "pot", "tureen", "soup bowl", "mixing bowl", "serving bowl", "plate"},
    "banana": {"fruit", "plantain", "yellow fruit", "tropical fruit", "edible fruit", "berry", "dessert fruit", "snack fruit", "exotic fruit", "sweet fruit"},
    "apple": {"fruit", "pome", "orchard fruit", "eating apple", "dessert apple", "crabapple", "cider fruit", "apple tree fruit", "fall fruit", "harvest fruit"},
    "sandwich": {"sub", "hoagie", "sarnie", "butty", "hero", "grinder", "po' boy", "panini", "club sandwich", "finger sandwich"},
    "orange": {"fruit", "citrus", "citrus fruit", "orange fruit", "tangerine", "mandarin", "navel orange", "blood orange", "clementine", "satsuma"},
    "broccoli": {"vegetable", "cruciferous vegetable", "green vegetable", "florets", "broccolini", "broccoli raab", "edible plant", "cauliflower relative", "brassica", "nutritious vegetable"},
    "carrot": {"vegetable", "root vegetable", "edible root", "orange vegetable", "garden vegetable", "produce", "carrot stick", "baby carrot", "carrot top", "nutritious vegetable"},
    "hot dog": {"frankfurter", "sausage", "weenie", "wiener", "bratwurst", "link", "dog", "frank", "hotdog", "sandwich"},
    "pizza": {"pie", "flatbread", "slice", "cheese pizza", "pepperoni pizza", "margherita", "deep dish", "thin crust", "pizza pie", "pizzeria"},
    "donut": {"doughnut", "pastry", "cruller", "fritter", "dough ring", "glazed donut", "jelly donut", "cream-filled donut", "sugary treat", "bakery item"},
    "cake": {"dessert", "pastry", "gateau", "torte", "sponge cake", "layer cake", "birthday cake", "wedding cake", "cupcake", "baked good"},
    "chair": {"seat", "stool", "armchair", "recliner", "rocker", "lawn chair", "dining chair", "office chair", "bench", "throne"},
    "couch": {"sofa", "settee", "divan", "chesterfield", "lounge", "loveseat", "sectional", "futon", "davenport", "daybed"},
    "potted plant": {"plant", "houseplant", "indoor plant", "container plant", "potted greenery", "potted flower", "potted shrub", "decorative plant", "potted flora", "garden plant"},
    "bed": {"bunk", "sleeping place", "mattress", "cot", "sleeping pad", "hammock", "loft bed", "futon", "bedding", "platform bed"},
    "dining table": {"table", "dinner table", "kitchen table", "eating table", "breakfast table", "banquet table", "dining room table", "wooden table", "glass table", "round table"},
    "toilet": {"lavatory", "restroom", "WC", "loo", "john", "commode", "bathroom", "water closet", "privy", "throne"},
    "tv": {"television", "television set", "TV set", "idiot box", "boob tube", "telly", "tube", "screen", "flat screen", "HDTV"},
    "laptop": {"notebook", "portable computer", "notebook computer", "laptop computer", "ultrabook", "MacBook", "Chromebook", "netbook", "workstation", "PC"},
    "mouse": {"computer mouse", "pointing device", "input device", "wireless mouse", "optical mouse", "scroll mouse", "trackball", "cursor control", "USB mouse", "peripheral"},
    "remote": {"remote control", "clicker", "controller", "TV remote", "universal remote", "wireless remote", "infrared remote", "remote device", "remote unit", "handheld remote"},
    "keyboard": {"keypad", "typewriter keyboard", "computer keyboard", "musical keyboard", "QWERTY", "mechanical keyboard", "wireless keyboard", "gaming keyboard", "piano keyboard", "input device"},
    "cell phone": {"mobile phone", "cellular phone", "smartphone", "handset", "iPhone", "Android phone", "mobile device", "cellular device", "cellular handset", "mobile"},
    "microwave": {"microwave oven", "microwave appliance", "cooking microwave", "microwave machine", "kitchen microwave", "countertop microwave", "microwave cooker", "microwave unit", "microwave equipment", "microwave device"},
    "oven": {"stove", "range", "cooker", "kitchen oven", "baking oven", "cooking oven", "electric oven", "gas oven", "microwave oven", "convection oven"},
    "toaster": {"toasting appliance", "bread toaster", "toasting machine", "toast maker", "toasting device", "kitchen toaster", "electric toaster", "toasting equipment", "pop-up toaster", "countertop toaster"},
    "sink": {"basin", "washbasin", "washbowl", "lavatory", "wash basin", "kitchen sink", "bathroom sink", "utility sink", "wash area", "cleaning basin"},
    "refrigerator": {"fridge", "cooler", "icebox", "refrigeration unit", "freezer", "chiller", "refrigeration appliance", "cooling unit", "refrigeration device", "cold storage"},
    "book": {"volume", "tome", "publication", "work", "novel", "manual", "textbook", "guide", "paperback", "hardcover"},
    "clock": {"timepiece", "timekeeper", "watch", "chronometer", "alarm clock", "wall clock", "desk clock", "mantel clock", "digital clock", "grandfather clock"},
    "vase": {"urn", "container", "jar", "pot", "pitcher", "jug", "amphora", "bottle", "flowerpot", "flower holder"},
    "scissors": {"shears", "clippers", "snips", "cutters", "trimmers", "nippers", "scissortail", "cutting tool", "cutting device", "pair of scissors"},
    "teddy bear": {"stuffed animal", "plush bear", "cuddly toy", "soft toy", "teddy", "toy bear", "plush toy", "stuffed toy", "bear toy", "cuddle toy"},
    "hair drier": {"hair dryer", "blow dryer", "blow drier", "hair blower", "drying appliance", "hair drying machine", "blow drying machine", "handheld dryer", "salon dryer", "personal dryer"},
    "toothbrush": {"dental brush", "tooth brush", "oral brush", "dental hygiene tool", "teeth brush", "bristle brush", "electric toothbrush", "manual toothbrush", "dental care brush", "oral care brush"}
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

    # Create a list of tokens for easier manipulation
    tokens = [token.text for token in doc]

    # Check for single-word matches and handle negations
    for token in doc:
        singular_form = p.singular_noun(token.text) if p.singular_noun(token.text) else token.text

        # Check for negation
        if token.dep_ == "neg":
            head = token.head
            head_singular_form = p.singular_noun(head.text) if p.singular_noun(head.text) else head.text
            if head.text in categories_set or head_singular_form in categories_set:
                exclude_list.add(head.text if head.text in categories_set else head_singular_form)
        else:
            # Check if token is a synonym and map it to the main category
            if token.text in synonym_to_category:
                main_category = synonym_to_category[token.text]
                include_list.add(main_category)
            elif singular_form in synonym_to_category:
                main_category = synonym_to_category[singular_form]
                include_list.add(main_category)
            elif token.text in categories_set or singular_form in categories_set:
                include_list.add(token.text if token.text in categories_set else singular_form)

    # Check for multi-word categories
    for category in categories:
        category_tokens = category.split()
        for i in range(len(tokens) - len(category_tokens) + 1):
            if tokens[i:i+len(category_tokens)] == category_tokens:
                include_list.add(category)
                break

    # Subtract exclude_list from include_list to get the final list
    matching_categories = list(include_list - exclude_list)

    return matching_categories

# Example usage
input_text = input("Enter text: ")
matches = find_matching_categories(input_text)
print("Matching categories:", matches)

# a=0
# for i in categories:
#     matches = find_matching_categories(i)
#     if matches:
#         a+=1
# print(a)