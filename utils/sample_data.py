"""Sample SMS messages for demonstration"""

SPAM_SAMPLES = [
    "CONGRATULATIONS! You've won a $1000 Walmart gift card! Click here to claim now: http://bit.ly/claim123",
    "FREE ENTRY in 2 a weekly comp for a chance to win an ipod. Txt POD to 80182 to get entry!",
    "URGENT! Your account has been compromised. Call 0800-BANK-NOW immediately to verify your details.",
    "You have been selected for a FREE cruise! Limited time offer. Reply YES to confirm.",
    "WINNER!! As a valued customer, you have been selected to receive £900 cash prize! Call 09061701461",
    "Want to lose weight fast? Try our new miracle pill! 100% guaranteed results. Order now!",
    "Hot singles in your area! Click here to meet them tonight! No credit card required.",
    "Your mobile won a Nokia N95! To claim, call 087123456789 now! Expires in 24hrs!",
    "CASH LOAN available now! £100-£1000 in your account TODAY. Bad credit OK. Text LOAN to 88888",
    "Double your income working from home! No experience needed. Reply NOW for details.",
]

LEGITIMATE_SAMPLES = [
    "Hi! Are we still meeting for coffee at 3pm today?",
    "Don't forget to pick up milk on your way home. Thanks!",
    "The project deadline has been extended to next Friday. Let me know if you have questions.",
    "Mom's birthday party is this Saturday at 6pm. See you there!",
    "Your appointment with Dr. Smith is confirmed for Tuesday at 10am.",
    "Meeting rescheduled to 2pm in Conference Room B.",
    "Can you send me the report when you get a chance? Thanks!",
    "Happy birthday! Hope you have a wonderful day!",
    "The train is delayed by 15 minutes. I'll be there soon.",
    "Thanks for your help today. Really appreciate it!",
]

def get_sample_messages():
    """Get sample spam and legitimate messages"""
    return {
        'spam': SPAM_SAMPLES,
        'legitimate': LEGITIMATE_SAMPLES
    }
