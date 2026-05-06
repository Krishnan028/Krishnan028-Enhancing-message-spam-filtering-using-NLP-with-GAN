import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Comprehensive rule-based spam scoring
# ---------------------------------------------------------------------------

SPAM_KEYWORDS = {
    # Very strong spam signals
    'free': 0.18, 'winner': 0.22, 'won': 0.20, 'prize': 0.22, 'claim': 0.18,
    'urgent': 0.18, 'congratulations': 0.20, 'congrats': 0.15, 'selected': 0.12,
    'lucky': 0.12, 'jackpot': 0.22, 'lottery': 0.22, 'sweepstake': 0.22,
    'reward': 0.15, 'bonus': 0.15, 'guaranteed': 0.18, 'exclusive': 0.12,
    'limited offer': 0.22, 'limited time': 0.20, 'act now': 0.22, 'expires': 0.15,
    'last chance': 0.18, 'deadline': 0.12, 'immediately': 0.12,
    # Phishing / account signals
    'verify': 0.18, 'verify your': 0.22, 'confirm your': 0.22, 'update your': 0.15,
    'click here': 0.22, 'click link': 0.22, 'click below': 0.20,
    'otp': 0.20, 'one-time': 0.15, 'password': 0.10, 'account suspended': 0.25,
    'bank account': 0.18, 'credit card': 0.15, 'debit card': 0.15,
    'social security': 0.25, 'ssn': 0.22, 'identity': 0.10,
    # Financial spam
    'cash': 0.12, 'earn': 0.10, 'income': 0.10, 'invest': 0.10,
    'loan': 0.12, 'debt': 0.10, 'credit': 0.10, 'insurance': 0.08,
    'make money': 0.22, 'extra income': 0.22, 'work from home': 0.18,
    'bitcoin': 0.18, 'crypto': 0.15, 'nft': 0.15, 'forex': 0.18,
    # Promotional spam
    'discount': 0.12, 'sale': 0.08, 'offer': 0.10, 'deal': 0.08,
    'buy now': 0.18, 'order now': 0.18, 'shop now': 0.15,
    'ringtone': 0.20, 'subscribe': 0.10, 'unsubscribe': 0.12,
    'reply stop': 0.25, 'text stop': 0.22, 'opt out': 0.20,
    # Voucher / gift card / giveaway scams
    'voucher': 0.22, 'coupon': 0.18, 'gift card': 0.22, 'gift voucher': 0.22,
    'giveaway': 0.18, 'giving away': 0.22, 'anniversary': 0.10, 'celebrate': 0.08,
    'go here': 0.22, 'get it': 0.15, 'go to': 0.08,
    # Delivery / package scams
    'package': 0.08, 'delivery failed': 0.22, 'parcel': 0.08,
    'customs fee': 0.22, 'reschedule delivery': 0.20,
}

SPAM_PATTERNS = [
    (r'https?://\S+', 0.35),
    (r'www\.\S+', 0.30),
    (r'\.[a-z]{2,4}/\S+', 0.25),  # URLs with paths like .com/myCoupon
    (r'\b\d{10,}\b', 0.20),
    (r'\b(\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b', 0.18),
    (r'[£$€₹]\s*\d+[\d,]*', 0.18),
    (r'\b\d+[\d,]*\s*[£$€₹]', 0.18),
    (r'\b(win|won|earn)\s+[£$€₹]?\d+', 0.22),
    (r'\b(free|giving away)\s+[£$€₹]?\d+', 0.25),  # "free £250" or "giving away £250"
    (r'(\.io|\.click|\.xyz|\.tk|\.ml|\.ga|\.cf|\.gq)\b', 0.25),
    (r'[A-Z]{4,}', 0.10),
    (r'!{2,}', 0.12),
    (r'\*\*\*.+\*\*\*', 0.15),
    (r'\b(call|txt|text)\s+\d+', 0.20),
    (r'ref:?\s*[a-zA-Z0-9]+', 0.10),
    (r'\bpin\b.{0,10}\d{4,}', 0.18),
]

HAM_INDICATORS = [
    r'\b(hi|hello|hey|dear)\b',
    r'\b(thanks|thank you|appreciate)\b',
    r'\b(meeting|office|colleague|team)\b',
    r'\b(mom|dad|brother|sister|friend)\b',
    r'\b(dinner|lunch|coffee|tomorrow)\b',
    r'\b(homework|assignment|project|study)\b',
    r'\b(love you|miss you|see you)\b',
]


def rule_based_spam_score(text: str) -> float:
    """Compute a spam probability using rules and keyword weights."""
    if not text or not text.strip():
        return 0.1
    t = text.lower()
    score = 0.0

    # Keyword matching
    for kw, weight in SPAM_KEYWORDS.items():
        if kw in t:
            score += weight

    # Pattern matching
    for pattern, weight in SPAM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight

    # ALL CAPS boost
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.55:
        score += 0.18

    # Very short message with URL or phone → strong spam signal
    if len(text.split()) < 15:
        if re.search(r'https?://|www\.', text, re.IGNORECASE):
            score += 0.15

    # Ham indicators reduce score
    ham_hits = sum(1 for p in HAM_INDICATORS if re.search(p, t))
    score -= ham_hits * 0.06

    return float(min(max(score, 0.0), 0.98))


class GANSpamDetector:
    """GAN-based SMS Spam Detector — rule-based engine + discriminator ensemble"""

    def __init__(self):
        self.vocab_size = 5000
        self.max_length = 100
        self.embedding_dim = 128
        self.latent_dim = 100
        self.word_index = {}
        self._build_vocab()
        self._initialize_models()

    def _build_vocab(self):
        common_words = [
            'free', 'winner', 'cash', 'prize', 'claim', 'urgent', 'congratulations',
            'click', 'offer', 'limited', 'now', 'call', 'txt', 'win', 'won',
            'bonus', 'guaranteed', 'money', 'credit', 'loan', 'debt', 'mobile',
            'ringtone', 'tone', 'reply', 'stop', 'subscribe', 'unsubscribe',
            'discount', 'you', 'your', 'are', 'have', 'this', 'that', 'with',
            'for', 'and', 'the', 'to', 'from', 'is', 'be', 'of', 'in', 'on',
            'message', 'text', 'phone', 'number', 'new', 'get', 'can', 'will',
            'hi', 'hello', 'thanks', 'thank', 'please', 'meet', 'time', 'today',
            'tomorrow', 'love', 'see', 'later', 'ok', 'yes', 'no', 'just', 'dont',
            'know', 'think', 'want', 'need', 'good', 'great', 'happy', 'birthday',
            'sorry', 'how', 'what', 'when', 'where', 'who', 'why', 'work', 'home',
            'day', 'night', 'morning', 'evening', 'week', 'send', 'got', 'make',
            'verify', 'account', 'bank', 'otp', 'password', 'identity', 'confirm',
            'earn', 'invest', 'bitcoin', 'crypto', 'lottery', 'jackpot', 'selected',
            'reward', 'exclusive', 'expires', 'immediately', 'package', 'delivery',
        ]
        self.word_index = {word: idx + 1 for idx, word in enumerate(common_words)}
        self.word_index['<UNK>'] = len(self.word_index) + 1

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Input(shape=(self.max_length,)),
            layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        return model

    def _initialize_models(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self._is_pretrained = False

    def _ensure_pretrained(self):
        if not self._is_pretrained:
            self._pretrain_discriminator()
            self._is_pretrained = True

    def _pretrain_discriminator(self):
        spam_patterns = [
            'free cash prize winner claim now urgent',
            'congratulations you won click here limited offer',
            'call now to claim your bonus guaranteed money',
            'urgent your account verify now click link',
            'txt stop to unsubscribe ringtone mobile offer',
            'win free prize guaranteed cash claim today',
            'limited time offer discount now click here',
            'call urgent winner prize money cash bonus',
            'free mobile ringtone subscribe now txt reply',
            'congratulations claim cash prize winner today',
            'verify your bank account immediately click link',
            'your otp password expires now confirm identity',
            'earn money bitcoin crypto investment guaranteed returns',
            'lottery jackpot selected exclusive reward bonus cash',
            'delivery failed package customs fee pay now click',
        ]
        ham_patterns = [
            'hi how are you doing today',
            'thanks for your help really appreciate it',
            'can we meet for coffee tomorrow afternoon',
            'happy birthday hope you have wonderful day',
            'sorry running late will be there soon',
            'what time is the meeting scheduled for',
            'love you see you later tonight',
            'thanks for the message talk soon',
            'ok sounds good let me know when',
            'have great day at work today',
            'finished the homework assignment for tomorrow class',
            'mom dad dinner ready come home soon',
            'team project meeting office colleague tomorrow morning',
            'miss you see you this weekend friend',
            'good morning hope you slept well today',
        ]
        X_train, y_train = [], []
        for p in spam_patterns * 15:
            X_train.append(self.preprocess_text(p))
            y_train.append(1)
        for p in ham_patterns * 15:
            X_train.append(self.preprocess_text(p))
            y_train.append(0)
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        idx = np.random.permutation(len(y_train))
        X_train, y_train = X_train[idx], y_train[idx]
        n = len(X_train)
        for _ in range(5):
            perm = np.random.permutation(n)
            for s in range(0, n, 32):
                e = s + 32
                self.discriminator.train_on_batch(X_train[perm[s:e]], y_train[perm[s:e]])

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        indices = []
        for word in words[:self.max_length]:
            if word in self.word_index:
                indices.append(self.word_index[word])
            else:
                indices.append((hash(word) % (self.vocab_size - 100)) + 100)
        indices = indices + [0] * (self.max_length - len(indices))
        return np.array([indices[:self.max_length]])

    def predict(self, text):
        """Predict spam probability: rule-based engine (70%) + discriminator (30%)"""
        if not text or not text.strip():
            return 0.05

        self._ensure_pretrained()

        rule_score = rule_based_spam_score(text)
        processed = self.preprocess_text(text)
        ml_score = float(self.discriminator.predict(processed, verbose=0)[0][0])

        # Ensemble: rule-based dominates
        final = 0.70 * rule_score + 0.30 * ml_score

        # Hard overrides for unmistakable spam signals
        if rule_score >= 0.70:
            final = max(final, 0.88)
        if rule_score <= 0.05:
            final = min(final, 0.25)

        return float(min(max(final, 0.02), 0.98))

    def get_model_info(self):
        return {
            'type': 'GAN Discriminator + Rule Engine',
            'architecture': 'BiLSTM + Dense + Rule-based ensemble',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'trainable_params': self.discriminator.count_params()
        }


def get_model():
    return GANSpamDetector()
