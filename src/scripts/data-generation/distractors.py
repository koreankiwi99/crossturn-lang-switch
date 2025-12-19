"""
Distractor sentences by language (irrelevant small talk / observations)
Used for generate_distractor.py to test noise filtering in context
"""

DISTRACTORS = {
    "zh": [  # Chinese
        "今天天气真不错。",  # The weather is really nice today.
        "我刚吃完午饭。",  # I just finished lunch.
        "最近工作很忙。",  # Work has been busy lately.
        "昨天我看了一部电影。",  # I watched a movie yesterday.
        "这个周末有什么计划吗？",  # Any plans for the weekend?
        "我在考虑换一部新手机。",  # I'm thinking about getting a new phone.
        "咖啡真的很提神。",  # Coffee is really refreshing.
        "我家的猫今天很可爱。",  # My cat was very cute today.
        "时间过得真快啊。",  # Time flies so fast.
        "我需要多运动了。",  # I need to exercise more.
        "最近学了一些新东西。",  # I learned something new recently.
        "早上的交通真堵。",  # Morning traffic was really bad.
        "我在想晚餐吃什么。",  # I'm thinking about what to eat for dinner.
        "这首歌很好听。",  # This song is really nice.
        "我喜欢这个季节。",  # I like this season.
        "周末想去爬山。",  # I want to go hiking this weekend.
        "我的朋友刚从旅行回来。",  # My friend just came back from a trip.
        "今天心情不错。",  # I'm in a good mood today.
        "我正在学习一门新语言。",  # I'm learning a new language.
        "最近睡眠质量不太好。",  # My sleep quality hasn't been great lately.
    ],
    "de": [  # German
        "Das Wetter ist heute wirklich schön.",  # The weather is really nice today.
        "Ich habe gerade zu Mittag gegessen.",  # I just finished lunch.
        "Die Arbeit war in letzter Zeit sehr stressig.",  # Work has been busy lately.
        "Gestern habe ich einen Film gesehen.",  # I watched a movie yesterday.
        "Hast du Pläne für das Wochenende?",  # Any plans for the weekend?
        "Ich überlege, mir ein neues Handy zu kaufen.",  # I'm thinking about getting a new phone.
        "Kaffee macht wirklich wach.",  # Coffee is really refreshing.
        "Meine Katze war heute sehr süß.",  # My cat was very cute today.
        "Die Zeit vergeht so schnell.",  # Time flies so fast.
        "Ich muss mehr Sport treiben.",  # I need to exercise more.
        "Ich habe kürzlich etwas Neues gelernt.",  # I learned something new recently.
        "Der Verkehr heute Morgen war schrecklich.",  # Morning traffic was really bad.
        "Ich überlege, was ich zum Abendessen essen soll.",  # I'm thinking about what to eat for dinner.
        "Dieses Lied gefällt mir sehr.",  # This song is really nice.
        "Ich mag diese Jahreszeit.",  # I like this season.
        "Am Wochenende möchte ich wandern gehen.",  # I want to go hiking this weekend.
        "Mein Freund ist gerade von einer Reise zurückgekommen.",  # My friend just came back from a trip.
        "Ich bin heute gut gelaunt.",  # I'm in a good mood today.
        "Ich lerne gerade eine neue Sprache.",  # I'm learning a new language.
        "Mein Schlaf war in letzter Zeit nicht so gut.",  # My sleep quality hasn't been great lately.
    ],
    "es": [  # Spanish
        "El clima está muy agradable hoy.",  # The weather is really nice today.
        "Acabo de terminar de almorzar.",  # I just finished lunch.
        "El trabajo ha estado muy ocupado últimamente.",  # Work has been busy lately.
        "Ayer vi una película.",  # I watched a movie yesterday.
        "¿Tienes planes para el fin de semana?",  # Any plans for the weekend?
        "Estoy pensando en comprar un teléfono nuevo.",  # I'm thinking about getting a new phone.
        "El café realmente me despierta.",  # Coffee is really refreshing.
        "Mi gato estuvo muy lindo hoy.",  # My cat was very cute today.
        "El tiempo pasa muy rápido.",  # Time flies so fast.
        "Necesito hacer más ejercicio.",  # I need to exercise more.
        "Aprendí algo nuevo recientemente.",  # I learned something new recently.
        "El tráfico esta mañana fue terrible.",  # Morning traffic was really bad.
        "Estoy pensando qué cenar.",  # I'm thinking about what to eat for dinner.
        "Esta canción me gusta mucho.",  # This song is really nice.
        "Me gusta esta temporada.",  # I like this season.
        "Este fin de semana quiero ir de excursión.",  # I want to go hiking this weekend.
        "Mi amigo acaba de volver de un viaje.",  # My friend just came back from a trip.
        "Hoy estoy de buen humor.",  # I'm in a good mood today.
        "Estoy aprendiendo un nuevo idioma.",  # I'm learning a new language.
        "Mi sueño no ha sido muy bueno últimamente.",  # My sleep quality hasn't been great lately.
    ],
    "ar": [  # Arabic
        "الطقس جميل جداً اليوم.",  # The weather is really nice today.
        "انتهيت للتو من تناول الغداء.",  # I just finished lunch.
        "العمل كان مشغولاً جداً مؤخراً.",  # Work has been busy lately.
        "شاهدت فيلماً بالأمس.",  # I watched a movie yesterday.
        "هل لديك خطط لعطلة نهاية الأسبوع؟",  # Any plans for the weekend?
        "أفكر في شراء هاتف جديد.",  # I'm thinking about getting a new phone.
        "القهوة منعشة حقاً.",  # Coffee is really refreshing.
        "قطتي كانت لطيفة جداً اليوم.",  # My cat was very cute today.
        "الوقت يمر بسرعة كبيرة.",  # Time flies so fast.
        "أحتاج إلى ممارسة المزيد من التمارين.",  # I need to exercise more.
        "تعلمت شيئاً جديداً مؤخراً.",  # I learned something new recently.
        "حركة المرور هذا الصباح كانت سيئة.",  # Morning traffic was really bad.
        "أفكر فيما سأتناوله على العشاء.",  # I'm thinking about what to eat for dinner.
        "هذه الأغنية جميلة جداً.",  # This song is really nice.
        "أحب هذا الموسم.",  # I like this season.
        "أريد الذهاب للمشي في الجبال هذا الأسبوع.",  # I want to go hiking this weekend.
        "صديقي عاد للتو من رحلة.",  # My friend just came back from a trip.
        "أنا في مزاج جيد اليوم.",  # I'm in a good mood today.
        "أتعلم لغة جديدة.",  # I'm learning a new language.
        "نومي لم يكن جيداً مؤخراً.",  # My sleep quality hasn't been great lately.
    ],
}
