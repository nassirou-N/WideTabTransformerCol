import re
import warnings
import numpy as np
import random
from gensim.models import Word2Vec
from collections import Counter, defaultdict
import hashlib
from sklearn.preprocessing import StandardScaler

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

# Enhanced operator sets with vulnerability-specific patterns
operators3 = {'<<=', '>>=', '**='}
operators2 = {
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', 
    '%=', '&=', '^=', '|=', '=>'
}
operators1 = {
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|', '=', ',', '?', ':', ';',
    '{', '}', '!', '~'
}

# Vulnerability-specific keywords and patterns
VULNERABILITY_KEYWORDS = {
    'reentrancy': {
        'call', 'value', 'send', 'transfer', 'withdraw', 'balance',
        'msg.sender', 'this.balance', 'call.value', 'external', 'payable',
        'delegatecall', 'callcode', 'selfdestruct'
    },
    'overflow': {
        'uint', 'int', 'SafeMath', 'add', 'sub', 'mul', 'div',
        'require', 'assert', 'overflow', 'underflow', 'unchecked'
    },
    'timestamp': {
        'now', 'block.timestamp', 'block.number', 'block.difficulty',
        'block.coinbase', 'timestamp', 'time', 'blockhash'
    }
}

CRITICAL_PATTERNS = {
    'dangerous_call': r'\.call\.value\(',
    'external_call': r'\.call\(',
    'balance_access': r'\.balance',
    'msg_sender': r'msg\.sender',
    'require_pattern': r'require\(',
    'assert_pattern': r'assert\(',
    'if_pattern': r'if\s*\(',
    'function_pattern': r'function\s+\w+',
    'modifier_pattern': r'modifier\s+\w+',
    'mapping_pattern': r'mapping\s*\(',
    'payable_pattern': r'payable',
    'external_pattern': r'external',
    'public_pattern': r'public',
    'private_pattern': r'private',
    'internal_pattern': r'internal',
    'delegatecall_pattern': r'\.delegatecall\(',
    'selfdestruct_pattern': r'selfdestruct\(',
    'state_variable': r'(uint|int|address|bool|string|bytes)\s+\w+'
}

class EnhancedFragmentVectorizer:
    def __init__(self, vector_length, use_augmentation=True):
        self.fragments = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0
        self.cnt = 1
        self.use_augmentation = use_augmentation
        
        # Enhanced features
        self.token_frequency = Counter()
        self.pattern_frequency = Counter()
        self.vocabulary = set()
        self.fragment_metadata = []
        
        # Statistical features
        self.fragment_lengths = []
        self.complexity_scores = []
        
        # Augmentation cache
        self.augmentation_cache = {}
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        
    @staticmethod
    def extract_critical_patterns(code_line):
        """Extract vulnerability-specific patterns from code"""
        patterns_found = []
        pattern_scores = {}
        
        for pattern_name, pattern_regex in CRITICAL_PATTERNS.items():
            matches = re.findall(pattern_regex, code_line, re.IGNORECASE)
            if matches:
                patterns_found.append(f"PATTERN_{pattern_name.upper()}")
                pattern_scores[pattern_name] = len(matches)
                
        return patterns_found, pattern_scores
    
    @staticmethod
    def calculate_complexity_score(tokens):
        """Calculate enhanced complexity score based on control structures and patterns"""
        complexity = 0
        control_keywords = {'if', 'else', 'for', 'while', 'require', 'assert', 'modifier'}
        state_changing = {'call', 'send', 'transfer', 'delegatecall', 'selfdestruct'}
        
        # Control flow complexity
        for token in tokens:
            if token.lower() in control_keywords:
                complexity += 1.5
            elif token in operators2 or token in operators3:
                complexity += 0.5
            elif token.lower() in state_changing:
                complexity += 2.0
                
        # Nesting depth estimation
        depth = 0
        max_depth = 0
        for token in tokens:
            if token == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif token == '}':
                depth = max(0, depth - 1)
                
        complexity += max_depth * 0.5
        
        return complexity
    
    @staticmethod
    def enhanced_tokenize(line):
        """Enhanced tokenization with better handling of Solidity patterns"""
        # First, extract and preserve critical patterns
        critical_patterns, pattern_scores = EnhancedFragmentVectorizer.extract_critical_patterns(line)
        
        tmp, w = [], []
        i = 0
        
        while i < len(line):
            # Skip whitespace but preserve structure
            if line[i] == ' ':
                if w:  # Only add non-empty words
                    tmp.append(''.join(w))
                    w = []
                # Skip multiple spaces
                while i < len(line) and line[i] == ' ':
                    i += 1
                continue
                
            # Handle 3-character operators
            elif i + 2 < len(line) and line[i:i + 3] in operators3:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i:i + 3])
                i += 3
                
            # Handle 2-character operators
            elif i + 1 < len(line) and line[i:i + 2] in operators2:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i:i + 2])
                i += 2
                
            # Handle 1-character operators
            elif line[i] in operators1:
                if w:
                    tmp.append(''.join(w))
                    w = []
                tmp.append(line[i])
                i += 1
                
            # Regular character
            else:
                w.append(line[i])
                i += 1
        
        # Add the last word if any
        if w:
            tmp.append(''.join(w))
        
        # Filter out empty tokens and add critical patterns
        tokens = [token for token in tmp if token.strip()]
        tokens.extend(critical_patterns)
        
        return tokens, pattern_scores

    def augment_fragment(self, fragment):
        """Data augmentation specific to smart contracts"""
        if not self.use_augmentation:
            return [fragment]
            
        augmented_fragments = [fragment]  # Always include original
        
        # 1. Variable renaming (preserve semantics)
        if random.random() < 0.4:
            renamed_fragment = self._rename_variables(fragment)
            if renamed_fragment != fragment:
                augmented_fragments.append(renamed_fragment)
        
        # 2. Comment injection (doesn't change logic)
        if random.random() < 0.3:
            commented_fragment = self._add_comments(fragment)
            augmented_fragments.append(commented_fragment)
        
        # 3. Whitespace variations
        if random.random() < 0.2:
            spaced_fragment = self._vary_whitespace(fragment)
            augmented_fragments.append(spaced_fragment)
        
        # 4. Equivalent statement reordering (safe only)
        if random.random() < 0.2:
            reordered_fragment = self._safe_reorder(fragment)
            if reordered_fragment != fragment:
                augmented_fragments.append(reordered_fragment)
        
        return augmented_fragments
    
    def _rename_variables(self, fragment):
        """Safely rename variables preserving code logic"""
        renamed_fragment = []
        var_mapping = {}
        var_counter = 0
        
        for line in fragment:
            new_line = line
            # Find variable declarations
            var_pattern = r'\b(uint|int|address|bool|string|bytes\d*)\s+(\w+)'
            matches = re.findall(var_pattern, line)
            
            for var_type, var_name in matches:
                if var_name not in var_mapping and not var_name.startswith('_'):
                    var_mapping[var_name] = f"var_{var_counter}"
                    var_counter += 1
            
            # Apply renaming
            for old_name, new_name in var_mapping.items():
                new_line = re.sub(r'\b' + old_name + r'\b', new_name, new_line)
            
            renamed_fragment.append(new_line)
        
        return renamed_fragment
    
    def _add_comments(self, fragment):
        """Add benign comments that don't affect logic"""
        commented_fragment = []
        comment_templates = [
            "// Security check",
            "// State update",
            "// External call",
            "// Validation"
        ]
        
        for i, line in enumerate(fragment):
            commented_fragment.append(line)
            # Add comment after certain patterns
            if any(pattern in line for pattern in ['require', 'assert', 'if']):
                if random.random() < 0.5:
                    commented_fragment.append(random.choice(comment_templates))
        
        return commented_fragment
    
    def _vary_whitespace(self, fragment):
        """Add whitespace variations"""
        spaced_fragment = []
        for line in fragment:
            if random.random() < 0.3:
                # Add spaces around operators
                for op in operators2:
                    line = line.replace(op, f" {op} ")
                # Clean up multiple spaces
                line = re.sub(r'\s+', ' ', line)
            spaced_fragment.append(line)
        return spaced_fragment
    
    def _safe_reorder(self, fragment):
        """Safely reorder independent statements"""
        # This is complex and needs careful analysis
        # For now, return original to avoid breaking code
        return fragment

    @staticmethod
    def tokenize_fragment(fragment):
        """Enhanced fragment tokenization with metadata extraction"""
        tokenized = []
        pattern_scores_total = defaultdict(int)
        function_regex = re.compile(r'function(\d)+')
        backwards_slice = False
        
        # Fragment-level metadata
        total_lines = len(fragment)
        has_external_calls = False
        has_state_changes = False
        vulnerability_indicators = 0
        max_nesting_depth = 0
        current_depth = 0
        
        for line in fragment:
            tokens, pattern_scores = EnhancedFragmentVectorizer.enhanced_tokenize(line)
            tokenized.extend(tokens)
            
            # Aggregate pattern scores
            for pattern, score in pattern_scores.items():
                pattern_scores_total[pattern] += score
            
            # Check for function pattern
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            
            # Analyze line for vulnerability indicators
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in VULNERABILITY_KEYWORDS['reentrancy']):
                vulnerability_indicators += 1
                if 'call' in line_lower and 'value' in line_lower:
                    has_external_calls = True
            
            if 'balance' in line_lower and ('=' in line or '+=' in line or '-=' in line):
                has_state_changes = True
            
            # Track nesting depth
            for token in tokens:
                if token == '{':
                    current_depth += 1
                    max_nesting_depth = max(max_nesting_depth, current_depth)
                elif token == '}':
                    current_depth = max(0, current_depth - 1)
        
        # Add structural tokens for better representation
        structural_tokens = [
            f"LINES_{min(total_lines, 20)}",
            f"VULN_INDICATORS_{min(vulnerability_indicators, 10)}",
            f"NESTING_DEPTH_{min(max_nesting_depth, 5)}"
        ]
        
        if has_external_calls:
            structural_tokens.append("HAS_EXTERNAL_CALLS")
        if has_state_changes:
            structural_tokens.append("HAS_STATE_CHANGES")
        
        # Add pattern frequency tokens
        for pattern, count in pattern_scores_total.items():
            if count > 0:
                structural_tokens.append(f"FREQ_{pattern}_{min(count, 5)}")
            
        tokenized.extend(structural_tokens)
        
        return tokenized, backwards_slice

    def add_fragment(self, fragment):
        """Enhanced fragment addition with metadata collection and augmentation"""
        # Apply augmentation if enabled
        if self.use_augmentation:
            augmented_fragments = self.augment_fragment(fragment)
        else:
            augmented_fragments = [fragment]
        
        for aug_fragment in augmented_fragments:
            tokenized_fragment, backwards_slice = self.tokenize_fragment(aug_fragment)
            
            # Calculate fragment statistics
            complexity = self.calculate_complexity_score(tokenized_fragment)
            self.complexity_scores.append(complexity)
            self.fragment_lengths.append(len(tokenized_fragment))
            
            # Update vocabulary and frequency counters
            self.vocabulary.update(tokenized_fragment)
            self.token_frequency.update(tokenized_fragment)
            
            # Store fragment with enhanced information
            fragment_info = {
                'tokens': tokenized_fragment,
                'backwards_slice': backwards_slice,
                'complexity': complexity,
                'length': len(tokenized_fragment),
                'fragment_id': self.cnt,
                'is_augmented': aug_fragment != fragment
            }
            
            self.fragments.append(tokenized_fragment)
            self.fragment_metadata.append(fragment_info)
            
            if backwards_slice:
                self.backward_slices += 1
            else:
                self.forward_slices += 1
                
            self.cnt += 1

    def create_enhanced_word2vec_model(self):
        """Create enhanced Word2Vec model with optimized parameters"""
        # Filter out very rare tokens
        filtered_fragments = []
        for fragment in self.fragments:
            filtered_tokens = [token for token in fragment 
                             if self.token_frequency[token] >= 3]  # Increased threshold
            if filtered_tokens:
                filtered_fragments.append(filtered_tokens)
        
        # Enhanced Word2Vec parameters for better stability
        model = Word2Vec(
            sentences=filtered_fragments,
            vector_size=self.vector_length,
            window=12,  # Increased window
            min_count=2,  # Increased minimum count
            workers=4,
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling
            negative=20,  # Increased negative samples
            epochs=50,  # More training epochs
            alpha=0.025,
            min_alpha=0.00001,
            seed=42,
            sample=1e-6  # Subsampling for frequent words
        )
        
        return model

    def vectorize(self, fragment):
        """Enhanced vectorization with improved sequence handling"""
        tokenized_fragment, backwards_slice = self.tokenize_fragment(fragment)
        
        # Initialize vector matrix
        vectors = np.zeros(shape=(100, self.vector_length), dtype=np.float32)
        
        # Get valid tokens (those in vocabulary)
        valid_tokens = [token for token in tokenized_fragment 
                       if token in self.embeddings.key_to_index]
        
        if not valid_tokens:
            return vectors
        
        # Enhanced positioning strategy with attention to important tokens
        importance_scores = self._calculate_token_importance(valid_tokens)
        
        if backwards_slice:
            # For backward slices, prioritize function signatures
            start_idx = max(0, 100 - len(valid_tokens))
            for i, (token, score) in enumerate(zip(valid_tokens[-100:], importance_scores[-100:])):
                if token in self.embeddings.key_to_index:
                    # Weight by importance
                    vectors[start_idx + i] = self.embeddings[token] * (1 + score * 0.5)
        else:
            # For forward slices, use importance-based positioning
            if len(valid_tokens) <= 100:
                for i, (token, score) in enumerate(zip(valid_tokens, importance_scores)):
                    if token in self.embeddings.key_to_index:
                        vectors[i] = self.embeddings[token] * (1 + score * 0.5)
            else:
                # Importance-based sampling for long fragments
                # Sort tokens by importance
                token_importance_pairs = list(zip(valid_tokens, importance_scores))
                token_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Take most important tokens
                important_tokens = token_importance_pairs[:40]
                # Add sequential tokens from beginning and end
                sequential_tokens = [(valid_tokens[i], importance_scores[i]) for i in range(30)]
                sequential_tokens += [(valid_tokens[i], importance_scores[i]) 
                                    for i in range(len(valid_tokens)-30, len(valid_tokens))]
                
                # Combine and remove duplicates
                all_tokens = important_tokens + sequential_tokens
                seen = set()
                unique_tokens = []
                for token, score in all_tokens:
                    if token not in seen and len(unique_tokens) < 100:
                        seen.add(token)
                        unique_tokens.append((token, score))
                
                # Fill vectors
                for i, (token, score) in enumerate(unique_tokens[:100]):
                    if token in self.embeddings.key_to_index:
                        vectors[i] = self.embeddings[token] * (1 + score * 0.5)
        
        return vectors
    
    def _calculate_token_importance(self, tokens):
        """Calculate importance score for each token"""
        importance_scores = []
        
        for token in tokens:
            score = 0.0
            
            # Check if token is a vulnerability keyword
            for vuln_type, keywords in VULNERABILITY_KEYWORDS.items():
                if token.lower() in keywords:
                    score += 2.0
            
            # Check if token is part of a critical pattern
            if token.startswith("PATTERN_"):
                score += 3.0
            
            # Check if token is a structural indicator
            if token.startswith(("LINES_", "VULN_", "HAS_", "FREQ_")):
                score += 1.5
            
            # Control flow tokens
            if token.lower() in {'if', 'else', 'while', 'for', 'require', 'assert'}:
                score += 1.0
            
            # State-changing operations
            if token.lower() in {'call', 'send', 'transfer', 'delegatecall'}:
                score += 2.5
            
            importance_scores.append(score)
        
        # Normalize scores
        max_score = max(importance_scores) if importance_scores else 1.0
        if max_score > 0:
            importance_scores = [s / max_score for s in importance_scores]
        
        return importance_scores

    def train_model(self):
        """Enhanced model training with vocabulary optimization"""
        print(f"Training enhanced Word2Vec model...")
        print(f"Total vocabulary size: {len(self.vocabulary)}")
        print(f"Total fragments (including augmented): {len(self.fragments)}")
        print(f"Average fragment length: {np.mean(self.fragment_lengths):.2f}")
        print(f"Average complexity score: {np.mean(self.complexity_scores):.2f}")
        
        # Create and train enhanced model
        model = self.create_enhanced_word2vec_model()
        self.embeddings = model.wv
        
        # Print vocabulary statistics
        print(f"Final vocabulary size: {len(self.embeddings.key_to_index)}")
        print(f"Most common tokens: {[token for token, count in self.token_frequency.most_common(10)]}")
        
        # Clean up memory
        del model
        
        # Verify critical tokens are in vocabulary
        critical_tokens = ['call', 'value', 'balance', 'msg.sender', 'require', 'transfer', 'send']
        missing_critical = [token for token in critical_tokens 
                          if token not in self.embeddings.key_to_index]
        if missing_critical:
            print(f"Warning: Critical tokens missing from vocabulary: {missing_critical}")
        else:
            print("All critical tokens present in vocabulary ✓")

    def get_vocabulary_stats(self):
        """Get detailed vocabulary statistics"""
        return {
            'total_tokens': len(self.vocabulary),
            'final_vocab_size': len(self.embeddings.key_to_index) if hasattr(self, 'embeddings') else 0,
            'avg_fragment_length': np.mean(self.fragment_lengths),
            'avg_complexity': np.mean(self.complexity_scores),
            'most_common_tokens': self.token_frequency.most_common(20),
            'total_fragments': len(self.fragments),
            'augmented_ratio': sum(1 for m in self.fragment_metadata if m.get('is_augmented', False)) / len(self.fragment_metadata) if self.fragment_metadata else 0
        }
    
    # In enhanced_fragment_vectorizer.py - Add inside EnhancedFragmentVectorizer class

    def compute_token_correlations(self):
        """
        Compute correlations between tokens and vulnerability labels
        """
        # Create token-document matrix
        n_fragments = len(self.fragments)
        unique_tokens = list(self.vocabulary)
        n_tokens = len(unique_tokens)
        
        # Build token frequency matrix
        token_matrix = np.zeros((n_fragments, n_tokens))
        labels = []
        
        for i, (fragment, metadata) in enumerate(zip(self.fragments, self.fragment_metadata)):
            for token in fragment:
                if token in unique_tokens:
                    j = unique_tokens.index(token)
                    token_matrix[i, j] += 1
            # Get label from metadata
            labels.append(metadata.get('label', 0))
        
        # Compute token-vulnerability correlations
        token_correlations = {}
        for j, token in enumerate(unique_tokens):
            if len(set(labels)) > 1:  # Check if we have both classes
                corr, _ = pearsonr(token_matrix[:, j], labels)
                token_correlations[token] = corr
            else:
                token_correlations[token] = 0.0
        
        # Sort by absolute correlation
        sorted_correlations = sorted(token_correlations.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)
        
        return {
            'token_correlations': token_correlations,
            'top_positive': [(t, c) for t, c in sorted_correlations if c > 0][:20],
            'top_negative': [(t, c) for t, c in sorted_correlations if c < 0][:20]
        }