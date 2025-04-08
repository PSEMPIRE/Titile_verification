from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from metaphone import doublemetaphone
import re
from .models import Title, TitleApplication
from django.utils import timezone
from translate import Translator
import itertools

# Define disallowed prefixes, suffixes, and words
DISALLOWED_PREFIXES = ["The", "India", "Bharat", "Samachar", "News", "Daily", "Weekly", "Times", "Express"]
DISALLOWED_SUFFIXES = ["News", "Times", "Express", "Today", "Patrika", "Samachar", "Journal", "Post"]
DISALLOWED_WORDS = ["Police", "Crime", "Corruption", "CBI", "CID", "Army", "Military", "Intelligence"]
PERIODICITY_TERMS = ["Daily", "Weekly", "Monthly", "Quarterly", "Biweekly", "Fortnightly", "Annual", "Yearly"]

# Common Indian languages for translation checks
LANGUAGES = ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "ur"]

# Normalize text
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Fuzzy match
def fuzzy_match(new_title, existing_titles, threshold=80):
    matches = []
    for title in existing_titles:
        score = fuzz.token_sort_ratio(new_title, title)
        if score >= threshold:
            matches.append((title, score))
    return matches

# Cosine similarity
def cosine_similarity_check(new_title, existing_titles, threshold=0.7):
    if not existing_titles:
        return []
    
    vectorizer = TfidfVectorizer().fit_transform([new_title] + existing_titles)
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    matches = [(existing_titles[i], round(cosine_sim[i], 2)) for i in range(len(existing_titles)) if cosine_sim[i] >= threshold]
    return matches

# Phonetic similarity
def phonetic_similarity_check(new_title, existing_titles):
    new_meta = set(doublemetaphone(new_title.lower()))
    matches = []
    for title in existing_titles:
        existing_meta = set(doublemetaphone(title.lower()))
        if new_meta & existing_meta:
            matches.append(title)
    return matches

# Check for disallowed words
def check_disallowed_words(title):
    title_words = normalize_text(title).split()
    found_words = [word for word in DISALLOWED_WORDS if word.lower() in title_words]
    return found_words

# Check for problematic prefixes/suffixes
def check_prefix_suffix(new_title, existing_titles, threshold=75):
    new_title_norm = normalize_text(new_title)
    new_title_words = new_title_norm.split()
    
    # Check if the title has disallowed prefixes/suffixes
    prefix_matches = []
    suffix_matches = []
    
    for title in existing_titles:
        title_norm = normalize_text(title)
        title_words = title_norm.split()
        
        # Check if the new title starts with a disallowed prefix and creates similarity
        for prefix in DISALLOWED_PREFIXES:
            if new_title_norm.startswith(prefix.lower()) and title_norm.startswith(prefix.lower()):
                # Compare the remaining parts of both titles
                new_remainder = ' '.join(new_title_words[1:]) if len(new_title_words) > 1 else ''
                title_remainder = ' '.join(title_words[1:]) if len(title_words) > 1 else ''
                
                if new_remainder and title_remainder:
                    similarity = fuzz.token_sort_ratio(new_remainder, title_remainder)
                    if similarity >= threshold:
                        prefix_matches.append((title, prefix, similarity))
        
        # Check if the new title ends with a disallowed suffix and creates similarity
        for suffix in DISALLOWED_SUFFIXES:
            if new_title_norm.endswith(suffix.lower()) and title_norm.endswith(suffix.lower()):
                # Compare the remaining parts of both titles
                new_remainder = ' '.join(new_title_words[:-1]) if len(new_title_words) > 1 else ''
                title_remainder = ' '.join(title_words[:-1]) if len(title_words) > 1 else ''
                
                if new_remainder and title_remainder:
                    similarity = fuzz.token_sort_ratio(new_remainder, title_remainder)
                    if similarity >= threshold:
                        suffix_matches.append((title, suffix, similarity))
    
    return prefix_matches, suffix_matches

# Check if title is a combination of existing titles
def check_title_combination(new_title, existing_titles, threshold=70):
    new_title_norm = normalize_text(new_title)
    combinations = []
    
    # Check pairs of existing titles
    for title1, title2 in itertools.combinations(existing_titles, 2):
        title1_norm = normalize_text(title1)
        title2_norm = normalize_text(title2)
        
        # Check if combining the two titles results in something similar to the new title
        combined_title = f"{title1_norm} {title2_norm}"
        similarity = fuzz.token_set_ratio(new_title_norm, combined_title)
        
        if similarity >= threshold:
            combinations.append((title1, title2, similarity))
        
        # Also check reverse order
        combined_title_rev = f"{title2_norm} {title1_norm}"
        similarity_rev = fuzz.token_set_ratio(new_title_norm, combined_title_rev)
        
        if similarity_rev >= threshold and (title1, title2, similarity) not in combinations:
            combinations.append((title2, title1, similarity_rev))
    
    return combinations

# Check for periodicity added to existing titles
def check_periodicity(new_title, existing_titles, threshold=75):
    new_title_norm = normalize_text(new_title)
    new_title_words = new_title_norm.split()
    periodicity_matches = []
    
    for title in existing_titles:
        title_norm = normalize_text(title)
        
        # Check if new title contains existing title + periodicity term or periodicity term + existing title
        for term in PERIODICITY_TERMS:
            term_lower = term.lower()
            
            # Remove periodicity term from new title
            if term_lower in new_title_words:
                new_title_without_term = ' '.join([word for word in new_title_words if word != term_lower])
                similarity = fuzz.token_sort_ratio(new_title_without_term, title_norm)
                
                if similarity >= threshold:
                    periodicity_matches.append((title, term, similarity))
    
    return periodicity_matches

# Check for similar meanings in other languages
def check_similar_meaning(new_title, existing_titles, threshold=80):
    similar_meanings = []
    
    try:
        # Translate the new title to English if it might be in another language
        translator = Translator(to_lang="en")
        translated_new_title = translator.translate(new_title)
        
        for title in existing_titles:
            # Translate existing title to English if needed
            translated_title = translator.translate(title)
            
            # Compare translated titles
            if translated_new_title != new_title or translated_title != title:
                similarity = fuzz.token_sort_ratio(translated_new_title, translated_title)
                
                if similarity >= threshold:
                    similar_meanings.append((title, similarity))
    except Exception:
        # If translation fails, skip this check
        pass
    
    return similar_meanings

@csrf_exempt
def submit_title(request):
    if request.method == 'POST':
        new_title = request.POST.get('title', '')
        normalized_title = normalize_text(new_title)
        
        # Fetch titles from the database
        db_titles = Title.objects.values_list('title', flat=True)
        
        # Also fetch pending applications to check against those too
        pending_titles = TitleApplication.objects.filter(
            status='pending',
            submission_date__gte=timezone.now() - timezone.timedelta(days=90)
        ).values_list('title', flat=True)
        
        all_titles = list(db_titles) + list(pending_titles)
        normalized_titles = [normalize_text(t) for t in all_titles]
        
        # Initialize results dictionary
        results = {
            'fuzzy_matches': [],
            'cosine_matches': [],
            'phonetic_matches': [],
            'disallowed_words': [],
            'prefix_matches': [],
            'suffix_matches': [],
            'title_combinations': [],
            'periodicity_matches': [],
            'similar_meanings': [],
            'probability': 100,
            'reasons': []
        }
        
        # Run basic similarity checks
        results['fuzzy_matches'] = fuzzy_match(normalized_title, normalized_titles)
        results['cosine_matches'] = cosine_similarity_check(normalized_title, normalized_titles)
        results['phonetic_matches'] = phonetic_similarity_check(normalized_title, normalized_titles)
        
        # Run guideline enforcement checks
        results['disallowed_words'] = check_disallowed_words(new_title)
        prefix_matches, suffix_matches = check_prefix_suffix(new_title, all_titles)
        results['prefix_matches'] = prefix_matches
        results['suffix_matches'] = suffix_matches
        results['title_combinations'] = check_title_combination(new_title, all_titles)
        results['periodicity_matches'] = check_periodicity(new_title, all_titles)
        results['similar_meanings'] = check_similar_meaning(new_title, all_titles)
        
        # Collect rejection reasons
        if results['fuzzy_matches']:
            results['reasons'].append(f"Fuzzy match found with: {', '.join([f'{m[0]} ({m[1]}%)' for m in results['fuzzy_matches']])}")
        
        if results['cosine_matches']:
            results['reasons'].append(f"Cosine similarity match found with: {', '.join([f'{m[0]} ({m[1]})' for m in results['cosine_matches']])}")
        
        if results['phonetic_matches']:
            results['reasons'].append(f"Phonetically similar titles found: {', '.join(results['phonetic_matches'])}")
        
        if results['disallowed_words']:
            results['reasons'].append(f"Contains disallowed words: {', '.join(results['disallowed_words'])}")
        
        if results['prefix_matches']:
            results['reasons'].append(f"Uses disallowed prefix similar to existing title(s): {', '.join([f'{m[0]} (Prefix: {m[1]}, {m[2]}%)' for m in results['prefix_matches']])}")
        
        if results['suffix_matches']:
            results['reasons'].append(f"Uses disallowed suffix similar to existing title(s): {', '.join([f'{m[0]} (Suffix: {m[1]}, {m[2]}%)' for m in results['suffix_matches']])}")
        
        if results['title_combinations']:
            results['reasons'].append(f"Appears to be a combination of existing titles: {', '.join([f'{m[0]} + {m[1]} ({m[2]}%)' for m in results['title_combinations']])}")
        
        if results['periodicity_matches']:
            results['reasons'].append(f"Adds periodicity term to existing title: {', '.join([f'{m[0]} + {m[1]} ({m[2]}%)' for m in results['periodicity_matches']])}")
        
        if results['similar_meanings']:
            results['reasons'].append(f"Has similar meaning to existing title in another language: {', '.join([f'{m[0]} ({m[1]}%)' for m in results['similar_meanings']])}")
        
        # Calculate uniqueness score
        total_checks = 9  # Total number of check types
        matched_checks = sum(bool(results[key]) for key in 
        [
            'fuzzy_matches', 
            'cosine_matches', 
            'phonetic_matches', 
            'disallowed_words',
            'prefix_matches', 
            'suffix_matches', 
            'title_combinations', 
            'periodicity_matches', 
            'similar_meanings'
        ])
        
        # Calculate probability as per requirements
        if matched_checks > 0:
            # Find highest similarity score
            similarity_scores = []
            
            if results['fuzzy_matches']:
                similarity_scores.extend([m[1] for m in results['fuzzy_matches']])
                
            if results['cosine_matches']:
                similarity_scores.extend([m[1] * 100 for m in results['cosine_matches']])  # Convert to percentage
                
            if results['prefix_matches']:
                similarity_scores.extend([m[2] for m in results['prefix_matches']])
                
            if results['suffix_matches']:
                similarity_scores.extend([m[2] for m in results['suffix_matches']])
                
            if results['title_combinations']:
                similarity_scores.extend([m[2] for m in results['title_combinations']])
                
            if results['periodicity_matches']:
                similarity_scores.extend([m[2] for m in results['periodicity_matches']])
                
            if results['similar_meanings']:
                similarity_scores.extend([m[1] for m in results['similar_meanings']])
            
            # If we have any similarity scores, use the highest to calculate probability
            if similarity_scores:
                highest_similarity = max(similarity_scores)
                results['probability'] = max(0, 100 - highest_similarity)  # As per requirements
            else:
                # If we have matches but no scores (e.g., just phonetic matches or disallowed words)
                results['probability'] = max(0, 100 - (matched_checks / total_checks * 100))
        
        # If we have rejection reasons, title is rejected
        is_rejected = bool(results['reasons'])
        
        if is_rejected:
            # Save application with rejected status
            TitleApplication.objects.create(
                title=new_title,
                status='rejected',
                rejection_reason='; '.join(results['reasons']),
                submission_date=timezone.now(),
                verification_probability=results['probability']
            )
            
            # Render response with rejection details
            return render(request, 'title_verification/submit_title.html', {
                'message': '❌ Rejected: ' + ' '.join(results['reasons']),
                'title': new_title,
                'probability': results['probability'],
                'results': results,
                'is_rejected': is_rejected
            })
        else:
            # Save title if it's unique
            Title.objects.create(title=new_title)
            
            # Also save as an accepted application
            TitleApplication.objects.create(
                title=new_title,
                status='accepted',
                submission_date=timezone.now(),
                verification_probability=100
            )
            
            # Render response with acceptance details
            return render(request, 'title_verification/submit_title.html', {
                'message': '✅ Title is unique and accepted!',
                'title': new_title,
                'probability': 100,
                'results': {k: [] for k in results.keys()},
                'is_rejected': is_rejected
            })
    
    # GET request - just render the form
    return render(request, 'title_verification/submit_title.html')

@csrf_exempt
def modify_and_resubmit(request):
    if request.method == 'POST':
        original_title = request.POST.get('original_title', '')
        modified_title = request.POST.get('modified_title', '')
        
        # Handle modified title submission
        # Here we simply redirect to the submit_title view with the modified title
        return render(request, 'title_verification/submit_title.html', {
            'title': modified_title,
            'original_title': original_title,
            'is_modification': True
        })
    
    return render(request, 'title_verification/modify_title.html')