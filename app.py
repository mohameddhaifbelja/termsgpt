import streamlit as st
from utils import count_n_tokens, split_text
from llm import summarize_results, llm_terms, MAX_TOKENS, COMPLETIION_TOKENS, term_prompt_len, summarize_prompt_len


def get_results(text):
    text = text.strip()
    if not text:
        return text

    # Extract information
    if count_n_tokens(text) + COMPLETIION_TOKENS + term_prompt_len < MAX_TOKENS:
        return llm_terms(text)

    sections = split_text(text=text,
                          max_tokens=MAX_TOKENS - COMPLETIION_TOKENS,
                          prompt_n_tokens=term_prompt_len)
    lengths = [count_n_tokens(txt) for txt in sections]
    terms_each_section = [llm_terms(txt) for txt in sections
                          ]
    result = ""
    for term in terms_each_section:
        nbr_tokens = count_n_tokens(result)
        if count_n_tokens(result) + count_n_tokens(term) + COMPLETIION_TOKENS + summarize_prompt_len < MAX_TOKENS:
            result += term

        else:
            result = summarize_results(result)
            result += term
    print(result)
    result = summarize_results(result)

    return result


# Set page title
st.title("Term GPT")

# Add input field for user to enter text
user_input = st.text_area("Terms And Agreements:")

# Process user input when button is clicked
if st.button("Process"):
    # Process user input using process_text function
    processed_text = get_results(user_input)
    # Display the processed text
    st.write(processed_text)
#
# if __name__ == "__main__":
#     text = """
#     TikTok
# How TikTok is supporting our community through COVID-19
#
# 1. Your Relationship With Us
# 2. Accepting the Terms
# 3. Changes to the Terms
# 4. Your Account with Us
# 5. Your Access to and Use of Our Services
# 6. Intellectual Property Rights
# 7. Content
# 8. Indemnity
# 9. EXCLUSION OF WARRANTIES
# 10. LIMITATION OF LIABILITY
# 11. Other Terms
# App Stores
# Contact Us.
# U.S.
#
# Terms of Service
# (If you are a user having your usual residence in the US)
#
# Last updated: February 2019
#
# 1. Your Relationship With Us
# Welcome to TikTok (the “Platform”), which is provided by TikTok Inc. in the United States (collectively such entities will be referred to as “TikTok”, “we” or “us”).
#
# You are reading the terms of service (the “Terms”), which govern the relationship and serve as an agreement between you and us and set forth the terms and conditions by which you may access and use the Platform and our related websites, services, applications, products and content (collectively, the “Services”). Access to certain Services or features of the Services (such as, by way of example and not limitation, the ability to submit or share User Content (defined below)) may be subject to age restrictions and not available to all users of the Services. Our Services are provided for private, non-commercial use. For purposes of these Terms, “you” and “your” means you as the user of the Services.
#
# The Terms form a legally binding agreement between you and us. Please take the time to read them carefully. If you are under age 18, you may only use the Services with the consent of your parent or legal guardian. Please be sure your parent or legal guardian has reviewed and discussed these Terms with you.
#
# ARBITRATION NOTICE FOR USERS IN THE UNITED STATES: THESE TERMS CONTAIN AN ARBITRATION CLAUSE AND A WAIVER OF RIGHTS TO BRING A CLASS ACTION AGAINST US. EXCEPT FOR CERTAIN TYPES OF DISPUTES MENTIONED IN THAT ARBITRATION CLAUSE, YOU AND TIKTOK AGREE THAT DISPUTES BETWEEN US WILL BE RESOLVED BY MANDATORY BINDING ARBITRATION, AND YOU AND TIKTOK WAIVE ANY RIGHT TO PARTICIPATE IN A CLASS-ACTION LAWSUIT OR CLASS-WIDE ARBITRATION.
#
# 2. Accepting the Terms
# By accessing or using our Services, you confirm that you can form a binding contract with TikTok, that you accept these Terms and that you agree to comply with them. Your access to and use of our Services is also subject to our Privacy Policy and Community Guidelines, the terms of which can be found directly on the Platform, or where the Platform is made available for download, on your mobile device’s applicable app store, and are incorporated herein by reference. By using the Services, you consent to the terms of the Privacy Policy.
#
# If you are accessing or using the Services on behalf of a business or entity, then (a) “you” and “your” includes you and that business or entity, (b) you represent and warrant that you are an authorized representative of the business or entity with the authority to bind the entity to these Terms, and that you agree to these Terms on the entity’s behalf, and (c) your business or entity is legally and financially responsible for your access or use of the Services as well as for the access or use of your account by others affiliated with your entity, including any employees, agents or contractors.
#
# You can accept the Terms by accessing or using our Services. You understand and agree that we will treat your access or use of the Services as acceptance of the Terms from that point onwards.
#
# You should print off or save a local copy of the Terms for your records.
#
# 3. Changes to the Terms
#     We amend these Terms from time to time, for instance when we update the functionality of our Services, when we combine multiple apps or services operated by us or our affiliates into a single combined service or app, or when there are regulatory changes. We will use commercially reasonable efforts to generally notify all users of any material changes to these Terms, such as through a notice on our Platform, however, you should look at the Terms regularly to check for such changes. We will also update the “Last Updated” date at the top of these Terms, which reflect the effective date of such Terms. Your continued access or use of the Services after the date of the new Terms constitutes your acceptance of the new Terms. If you do not agree to the new Terms, you must stop accessing or using the Services.
#
# 4. Your Account with Us
# To access or use some of our Services, you must create an account with us. When you create this account, you must provide accurate and up-to-date information. It is important that you maintain and promptly update your details and any other information you provide to us, to keep such information current and complete.
#
# It is important that you keep your account password confidential and that you do not disclose it to any third party. If you know or suspect that any third party knows your password or has accessed your account, you must notify us immediately at: https://www.tiktok.com/legal/report/feedback.
#
# You agree that you are solely responsible (to us and to others) for the activity that occurs under your account.
#
# We reserve the right to disable your user account at any time, including if you have failed to comply with any of the provisions of these Terms, or if activities occur on your account which, in our sole discretion, would or might cause damage to or impair the Services or infringe or violate any third party rights, or violate any applicable laws or regulations.
#
# If you no longer want to use our Services again, and would like your account deleted, contact us at: https://www.tiktok.com/legal/report/feedback. We will provide you with further assistance and guide you through the process. Once you choose to delete your account, you will not be able to reactivate your account or retrieve any of the content or information you have added.
#
# 5. Your Access to and Use of Our Services
# Your access to and use of the Services is subject to these Terms and all applicable laws and regulations. You may not:
#
# access or use the Services if you are not fully able and legally competent to agree to these Terms or are authorized to use the Services by your parent or legal guardian;
# make unauthorised copies, modify, adapt, translate, reverse engineer, disassemble, decompile or create any derivative works of the Services or any content included therein, including any files, tables or documentation (or any portion thereof) or determine or attempt to determine any source code, algorithms, methods or techniques embodied by the Services or any derivative works thereof;
# distribute, license, transfer, or sell, in whole or in part, any of the Services or any derivative works thereof
# market, rent or lease the Services for a fee or charge, or use the Services to advertise or perform any commercial solicitation;
# use the Services, without our express written consent, for any commercial or unauthorized purpose, including communicating or facilitating any commercial advertisement or solicitation or spamming;
# interfere with or attempt to interfere with the proper working of the Services, disrupt our website or any networks connected to the Services, or bypass any measures we may use to prevent or restrict access to the Services;
# incorporate the Services or any portion thereof into any other program or product. In such case, we reserve the right to refuse service, terminate accounts or limit access to the Services in our sole discretion;
# use automated scripts to collect information from or otherwise interact with the Services;
# impersonate any person or entity, or falsely state or otherwise misrepresent you or your affiliation with any person or entity, including giving the impression that any content you upload, post, transmit, distribute or otherwise make available emanates from the Services;
# intimidate or harass another, or promote sexually explicit material, violence or discrimination based on race, sex, religion, nationality, disability, sexual orientation or age;
# use or attempt to use another’s account, service or system without authorisation from TikTok, or create a false identity on the Services;
# use the Services in a manner that may create a conflict of interest or undermine the purposes of the Services, such as trading reviews with other users or writing or soliciting fake reviews;
# use the Services to upload, transmit, distribute, store or otherwise make available in any way: files that contain viruses, trojans, worms, logic bombs or other material that is malicious or technologically harmful;
# any unsolicited or unauthorised advertising, solicitations, promotional materials, “junk mail,” “spam,” “chain letters,” “pyramid schemes,” or any other prohibited form of solicitation;
# any private information of any third party, including addresses, phone numbers, email addresses, number and feature in the personal identity document (e.g., National Insurance numbers, passport numbers) or credit card numbers;
# any material which does or may infringe any copyright, trademark or other intellectual property or privacy rights of any other person;
# any material which is defamatory of any person, obscene, offensive, pornographic, hateful or inflammatory;
# any material that would constitute, encourage or provide instructions for a criminal offence, dangerous activities or self-harm;
# any material that is deliberately designed to provoke or antagonise people, especially trolling and bullying, or is intended to harass, harm, hurt, scare, distress, embarrass or upset people;
# any material that contains a threat of any kind, including threats of physical violence;
# any material that is racist or discriminatory, including discrimination on the basis of someone’s race, religion, age, gender, disability or sexuality;
# any answers, responses, comments, opinions, analysis or recommendations that you are not properly licensed or otherwise qualified to provide; or
# material that, in the sole judgment of TikTok, is objectionable or which restricts or inhibits any other person from using the Services, or which may expose TikTok, the Services or its users to any harm or liability of any type.
# In addition to the above, your access to and use of the Services must, at all times, be compliant with our Community Guidelines.
#
# We reserve the right, at any time and without prior notice, to remove or disable access to content at our discretion for any reason or no reason. Some of the reasons we may remove or disable access to content may include finding the content objectionable, in violation of these Terms or our Community Policy, or otherwise harmful to the Services or our users. Our automated systems analyze your content (including emails) to provide you personally relevant product features, such as customized search results, tailored advertising, and spam and malware detection. This analysis occurs as the content is sent, received, and when it is stored.
#
# 6. Intellectual Property Rights
# We respect intellectual property rights and ask you to do the same. As a condition of your access to and use of the Services, you agree to the terms of the Copyright Policy.
#
# 7. Content
# TikTok Content
# As between you and TikTok, all content, software, images, text, graphics, illustrations, logos, patents, trademarks, service marks, copyrights, photographs, audio, videos, music on and “look and feel” of the Services, and all intellectual property rights related thereto (the “TikTok Content”), are either owned or licensed by TikTok, it being understood that you or your licensors will own any User Content (as defined below) you upload or transmit through the Services. Use of the TikTok Content or materials on the Services for any purpose not expressly permitted by these Terms is strictly prohibited. Such content may not be downloaded, copied, reproduced, distributed, transmitted, broadcast, displayed, sold, licensed or otherwise exploited for any purpose whatsoever without our or, where applicable, our licensors’ prior written consent. We and our licensors reserve all rights not expressly granted in and to their content.
#
# You acknowledge and agree that we may generate revenues, increase goodwill or otherwise increase our value from your use of the Services, including, by way of example and not limitation, through the sale of advertising, sponsorships, promotions, usage data and Gifts (defined below), and except as specifically permitted by us in these Terms or in another agreement you enter into with us, you will have no right to share in any such revenue, goodwill or value whatsoever. You further acknowledge that, except as specifically permitted by us in these Terms or in another agreement you enter into with us, you (i) have no right to receive any income or other consideration from any User Content (defined below) or your use of any musical works, sound recordings or audiovisual clips made available to you on or through the Services, including in any User Content created by you, and (ii) are prohibited from exercising any rights to monetize or obtain consideration from any User Content within the Services or on any third party service ( e.g. , you cannot claim User Content that has been uploaded to a social media platform such as YouTube for monetization).
#
# Subject to the terms and conditions of the Terms, you are hereby granted a non-exclusive, limited, non-transferable, non-sublicensable, revocable, worldwide license to access and use the Services, including to download the Platform on a permitted device, and to access the TIkTok Content solely for your personal, non-commercial use through your use of the Services and solely in compliance with these Terms. TikTok reserves all rights not expressly granted herein in the Services and the TikTok Content. You acknowledge and agree that TikTok may terminate this license at any time for any reason or no reason.
#
# NO RIGHTS ARE LICENSED WITH RESPECT TO SOUND RECORDINGS AND THE MUSICAL WORKS EMBODIED THEREIN THAT ARE MADE AVAILABLE FROM OR THROUGH THE SERVICE.
#
# You acknowledge and agree that when you view content provided by others on the Services, you are doing so at your own risk. The content on our Services is provided for general information only. It is not intended to amount to advice on which you should rely. You must obtain professional or specialist advice before taking, or refraining from, any action on the basis of the content on our Services.
#
# We make no representations, warranties or guarantees, whether express or implied, that any TikTok Content (including User Content) is accurate, complete or up to date. Where our Services contain links to other sites and resources provided by third parties, these links are provided for your information only. We have no control over the contents of those sites or resources. Such links should not be interpreted as approval by us of those linked websites or information you may obtain from them. You acknowledge that we have no obligation to pre-screen, monitor, review, or edit any content posted by you and other users on the Services (including User Content).
#
# User-Generated Content
# Users of the Services may be permitted to upload, post or transmit (such as via a stream) or otherwise make available content through the Services including, without limitation, any text, photographs, user videos, sound recordings and the musical works embodied therein, including videos that incorporate locally stored sound recordings from your personal music library and ambient noise (“User Content”). Users of the Services may also extract all or any portion of User Content created by another user to produce additional User Content, including collaborative User Content with other users, that combine and intersperse User Content generated by more than one user. Users of the Services may also overlay music, graphics, stickers, Virtual Items (as defined and further explained Virtual Items Policy) and other elements provided by TikTok (“TikTok Elements”) onto this User Content and transmit this User Content through the Services. The information and materials in the User Content, including User Content that includes TikTok Elements, have not been verified or approved by us. The views expressed by other users on the Services (including through use of the virtual gifts) do not represent our views or values.
#
# Whenever you access or use a feature that allows you to upload or transmit User Content through the Services (including via certain third party social media platforms such as Instagram, Facebook, YouTube, Twitter), or to make contact with other users of the Services, you must comply with the standards set out at “Your Access to and Use of Our Services” above. You may also choose to upload or transmit your User Content, including User Content that includes TikTok Elements, on sites or platforms hosted by third parties. If you decide to do this, you must comply with their content guidelines as well as with the standards set out at “Your Access to and Use of Our Services” above. As noted above, these features may not be available to all users of the Services, and we have no liability to you for limiting your right to certain features of the Services.
#
# You warrant that any such contribution does comply with those standards, and you will be liable to us and indemnify us for any breach of that warranty. This means you will be responsible for any loss or damage we suffer as a result of your breach of warranty.
#
# Any User Content will be considered non-confidential and non-proprietary. You must not post any User Content on or through the Services or transmit to us any User Content that you consider to be confidential or proprietary. When you submit User Content through the Services, you agree and represent that you own that User Content, or you have received all necessary permissions, clearances from, or are authorised by, the owner of any part of the content to submit it to the Services, to transmit it from the Services to other third party platforms, and/or adopt any third party content.
#
# If you only own the rights in and to a sound recording, but not to the underlying musical works embodied in such sound recordings, then you must not post such sound recordings to the Services unless you have all permissions, clearances from, or are authorised by, the owner of any part of the content to submit it to the Services
#
# You or the owner of your User Content still own the copyright in User Content sent to us, but by submitting User Content via the Services, you hereby grant us an unconditional irrevocable, non-exclusive, royalty-free, fully transferable, perpetual worldwide licence to use, modify, adapt, reproduce, make derivative works of, publish and/or transmit, and/or distribute and to authorise other users of the Services and other third-parties to view, access, use, download, modify, adapt, reproduce, make derivative works of, publish and/or transmit your User Content in any format and on any platform, either now known or hereinafter invented.
#
# You further grant us a royalty-free license to use your user name, image, voice, and likeness to identify you as the source of any of your User Content; provided, however, that your ability to provide an image, voice, and likeness may be subject to limitations due to age restrictions.
#
# For the avoidance of doubt, the rights granted in the preceding paragraphs of this Section include, but are not limited to, the right to reproduce sound recordings (and make mechanical reproductions of the musical works embodied in such sound recordings), and publicly perform and communicate to the public sound recordings (and the musical works embodied therein), all on a royalty-free basis. This means that you are granting us the right to use your User Content without the obligation to pay royalties to any third party, including, but not limited to, a sound recording copyright owner (e.g., a record label), a musical work copyright owner (e.g., a music publisher), a performing rights organization (e.g., ASCAP, BMI, SESAC, etc.) (a “PRO”), a sound recording PRO (e.g., SoundExchange), any unions or guilds, and engineers, producers or other royalty participants involved in the creation of User Content.
#
# Specific Rules for Musical Works and for Recording Artists. If you are a composer or author of a musical work and are affiliated with a PRO, then you must notify your PRO of the royalty-free license you grant through these Terms in your User Content to us. You are solely responsible for ensuring your compliance with the relevant PRO’s reporting obligations. If you have assigned your rights to a music publisher, then you must obtain the consent of such music publisher to grant the royalty-free license(s) set forth in these Terms in your User Content or have such music publisher enter into these Terms with us. Just because you authored a musical work (e.g., wrote a song) does not mean you have the right to grant us the licenses in these Terms. If you are a recording artist under contract with a record label, then you are solely responsible for ensuring that your use of the Services is in compliance with any contractual obligations you may have to your record label, including if you create any new recordings through the Services that may be claimed by your label.
#
# Through-To-The-Audience Rights. All of the rights you grant in your User Content in these Terms are provided on a through-to-the-audience basis, meaning the owners or operators of third party services will not have any separate liability to you or any other third party for User Content posted or used on such third party service via the Services.
#
# Waiver of Rights to User Content. By posting User Content to or through the Services, you waive any rights to prior inspection or approval of any marketing or promotional materials related to such User Content. You also waive any and all rights of privacy, publicity, or any other rights of a similar nature in connection with your User Content, or any portion thereof. To the extent any moral rights are not transferable or assignable, you hereby waive and agree never to assert any and all moral rights, or to support, maintain or permit any action based on any moral rights that you may have in or with respect to any User Content you Post to or through the Services.
#
# We also have the right to disclose your identity to any third party who is claiming that any User Content posted or uploaded by you to our Services constitutes a violation of their intellectual property rights, or of their right to privacy.
#
# We, or authorised third parties, reserve the right to cut, crop, edit or refuse to publish, your content at our or their sole discretion. We have the right to remove, disallow, block or delete any posting you make on our Services if, in our opinion, your post does not comply with the content standards set out at “Your Access to and Use of Our Services”above. In addition, we have the right – but not the obligation – in our sole discretion to remove, disallow, block or delete any User Content (i) that we consider to violate these Terms, or (ii) in response to complaints from other users or third parties, with or without notice and without any liability to you. As a result, we recommend that you save copies of any User Content that you post to the Services on your personal device(s) in the event that you want to ensure that you have permanent access to copies of such User Content. We do not guarantee the accuracy, integrity, appropriateness or quality of any User Content, and under no circumstances will we be liable in any way for any User Content.
#
# You control whether your User Content is made publicly available on the Services to all other users of the Services or only available to people you approve. To restrict access to your User Content, you should select the privacy setting available within the Platform.
#
# We accept no liability in respect of any content submitted by users and published by us or by authorised third parties.
#
# If you wish to file a complaint about information or materials uploaded by other users, contact us at: https://www.tiktok.com/legal/report/feedback.
#
# TikTok takes reasonable measures to expeditiously remove from our Services any infringing material that we become aware of.It is TikTok’s policy, in appropriate circumstances and at its discretion, to disable or terminate the accounts of users of the Services who repeatedly infringe copyrights or intellectual property rights of others.
#
# While our own staff is continually working to develop and evaluate our own product ideas and features, we pride ourselves on paying close attention to the interests, feedback, comments, and suggestions we receive from the user community. If you choose to contribute by sending us or our employees any ideas for products, services, features, modifications, enhancements, content, refinements, technologies, content offerings (such as audio, visual, games, or other types of content), promotions, strategies, or product/feature names, or any related documentation, artwork, computer code, diagrams, or other materials (collectively “Feedback”), then regardless of what your accompanying communication may say, the following terms will apply, so that future misunderstandings can be avoided. Accordingly, by sending Feedback to us, you agree that:
#
# TikTok has no obligation to review, consider, or implement your Feedback, or to return to you all or part of any Feedback for any reason;
#
# Feedback is provided on a non-confidential basis, and we are not under any obligation to keep any Feedback you send confidential or to refrain from using or disclosing it in any way; and
#
# You irrevocably grant us perpetual and unlimited permission to reproduce, distribute, create derivative works of, modify, publicly perform (including on a through-to-the-audience basis), communicate to the public, make available, publicly display, and otherwise use and exploit the Feedback and derivatives thereof for any purpose and without restriction, free of charge and without attribution of any kind, including by making, using, selling, offering for sale, importing, and promoting commercial products and services that incorporate or embody Feedback, whether in whole or in part, and whether as provided or as modified.
#
# 8. Indemnity
# You agree to defend, indemnify, and hold harmless TikTok, its parents, subsidiaries, and affiliates, and each of their respective officers, directors, employees, agents and advisors from any and all claims, liabilities, costs, and expenses, including, but not limited to, attorneys’ fees and expenses, arising out of a breach by you or any user of your account of these Terms or arising out of a breach of your obligations, representation and warranties under these Terms.
#
# 9. EXCLUSION OF WARRANTIES
# NOTHING IN THESE TERMS SHALL AFFECT ANY STATUTORY RIGHTS THAT YOU CANNOT CONTRACTUALLY AGREE TO ALTER OR WAIVE AND ARE LEGALLY ALWAYS ENTITLED TO AS A CONSUMER.
#
# THE SERVICES ARE PROVIDED “AS IS” AND WE MAKE NO WARRANTY OR REPRESENTATION TO YOU WITH RESPECT TO THEM. IN PARTICULAR WE DO NOT REPRESENT OR WARRANT TO YOU THAT:
#
# YOUR USE OF THE SERVICES WILL MEET YOUR REQUIREMENTS;
# YOUR USE OF THE SERVICES WILL BE UNINTERRUPTED, TIMELY, SECURE OR FREE FROM ERROR;
# ANY INFORMATION OBTAINED BY YOU AS A RESULT OF YOUR USE OF THE SERVICES WILL BE ACCURATE OR RELIABLE; AND
# DEFECTS IN THE OPERATION OR FUNCTIONALITY OF ANY SOFTWARE PROVIDED TO YOU AS PART OF THE SERVICES WILL BE CORRECTED.
# NO CONDITIONS, WARRANTIES OR OTHER TERMS (INCLUDING ANY IMPLIED TERMS AS TO SATISFACTORY QUALITY, FITNESS FOR PURPOSE OR CONFORMANCE WITH DESCRIPTION) APPLY TO THE SERVICES EXCEPT TO THE EXTENT THAT THEY ARE EXPRESSLY SET OUT IN THE TERMS. WE MAY CHANGE, SUSPEND, WITHDRAW OR RESTRICT THE AVAILABILITY OF ALL OR ANY PART OF OUR PLATFORM FOR BUSINESS AND OPERATIONAL REASONS AT ANY TIME WITHOUT NOTICE
#
# 10. LIMITATION OF LIABILITY
# NOTHING IN THESE TERMS SHALL EXCLUDE OR LIMIT OUR LIABILITY FOR LOSSES WHICH MAY NOT BE LAWFULLY EXCLUDED OR LIMITED BY APPLICABLE LAW. THIS INCLUDES LIABILITY FOR DEATH OR PERSONAL INJURY CAUSED BY OUR NEGLIGENCE OR THE NEGLIGENCE OF OUR EMPLOYEES, AGENTS OR SUBCONTRACTORS AND FOR FRAUD OR FRAUDULENT MISREPRESENTATION.
#
# SUBJECT TO THE PARAGRAPH ABOVE, WE SHALL NOT BE LIABLE TO YOU FOR:
#
# (I) ANY LOSS OF PROFIT (WHETHER INCURRED DIRECTLY OR INDIRECTLY);
# (II) ANY LOSS OF GOODWILL;
# (III) ANY LOSS OF OPPORTUNITY;
# (IV) ANY LOSS OF DATA SUFFERED BY YOU; OR
# (V) ANY INDIRECT OR CONSEQUENTIAL LOSSES WHICH MAY BE INCURRED BY YOU. ANY OTHER LOSS WILL BE LIMITED TO THE AMOUNT PAID BY YOU TO TIKTOK WITHIN THE LAST 12 MONTHS.
# ANY LOSS OR DAMAGE WHICH MAY BE INCURRED BY YOU AS A RESULT OF:
#
# ANY RELIANCE PLACED BY YOU ON THE COMPLETENESS, ACCURACY OR EXISTENCE OF ANY ADVERTISING, OR AS A RESULT OF ANY RELATIONSHIP OR TRANSACTION BETWEEN YOU AND ANY ADVERTISER OR SPONSOR WHOSE ADVERTISING APPEARS ON THE SERVICE;
# ANY CHANGES WHICH WE MAY MAKE TO THE SERVICES, OR FOR ANY PERMANENT OR TEMPORARY CESSATION IN THE PROVISION OF THE SERVICES (OR ANY FEATURES WITHIN THE SERVICES);
# THE DELETION OF, CORRUPTION OF, OR FAILURE TO STORE, ANY CONTENT AND OTHER COMMUNICATIONS DATA MAINTAINED OR TRANSMITTED BY OR THROUGH YOUR USE OF THE SERVICES;
# YOUR FAILURE TO PROVIDE US WITH ACCURATE ACCOUNT INFORMATION; OR
# YOUR FAILURE TO KEEP YOUR PASSWORD OR ACCOUNT DETAILS SECURE AND CONFIDENTIAL.
# PLEASE NOTE THAT WE ONLY PROVIDE OUR PLATFORM FOR DOMESTIC AND PRIVATE USE. YOU AGREE NOT TO USE OUR PLATFORM FOR ANY COMMERCIAL OR BUSINESS PURPOSES, AND WE HAVE NO LIABILITY TO YOU FOR ANY LOSS OF PROFIT, LOSS OF BUSINESS, LOSS OF GOODWILL OR BUSINESS REPUTATION, BUSINESS INTERRUPTION, OR LOSS OF BUSINESS OPPORTUNITY.
#
# IF DEFECTIVE DIGITAL CONTENT THAT WE HAVE SUPPLIED DAMAGES A DEVICE OR DIGITAL CONTENT BELONGING TO YOU AND THIS IS CAUSED BY OUR FAILURE TO USE REASONABLE CARE AND SKILL, WE WILL EITHER REPAIR THE DAMAGE OR PAY YOU COMPENSATION. HOWEVER, WE WILL NOT BE LIABLE FOR DAMAGE THAT YOU COULD HAVE AVOIDED BY FOLLOWING OUR ADVICE TO APPLY AN UPDATE OFFERED TO YOU FREE OF CHARGE OR FOR DAMAGE THAT WAS CAUSED BY YOU FAILING TO CORRECTLY FOLLOW INSTALLATION INSTRUCTIONS OR TO HAVE IN PLACE THE MINIMUM SYSTEM REQUIREMENTS ADVISED BY US.
#
# THESE LIMITATIONS ON OUR LIABILITY TO YOU SHALL APPLY WHETHER OR NOT WE HAVE BEEN ADVISED OF OR SHOULD HAVE BEEN AWARE OF THE POSSIBILITY OF ANY SUCH LOSSES ARISING.
#
# YOU ARE RESPONSIBLE FOR ANY MOBILE CHARGES THAT MAY APPLY TO YOUR USE OF OUR SERVICE, INCLUDING TEXT-MESSAGING AND DATA CHARGES. IF YOU’RE UNSURE WHAT THOSE CHARGES MAY BE, YOU SHOULD ASK YOUR SERVICE PROVIDER BEFORE USING THE SERVICE.
#
# TO THE FULLEST EXTENT PERMITTED BY LAW, ANY DISPUTE YOU HAVE WITH ANY THIRD PARTY ARISING OUT OF YOUR USE OF THE SERVICES, INCLUDING, BY WAY OF EXAMPLE AND NOT LIMITATION, ANY CARRIER, COPYRIGHT OWNER OR OTHER USER, IS DIRECTLY BETWEEN YOU AND SUCH THIRD PARTY, AND YOU IRREVOCABLY RELEASE US AND OUR AFFILIATES FROM ANY AND ALL CLAIMS, DEMANDS AND DAMAGES (ACTUAL AND CONSEQUENTIAL) OF EVERY KIND AND NATURE, KNOWN AND UNKNOWN, ARISING OUT OF OR IN ANY WAY CONNECTED WITH SUCH DISPUTES.
#
# 11. Other Terms
# Open Source.The Platform contains certain open source software. Each item of open source software is subject to its own applicable license terms, which can be found at Open Source Policy.
#
# Entire Agreement.These Terms constitute the whole legal agreement between you and TikTok and govern your use of the Services and completely replace any prior agreements between you and TikTok in relation to the Services.
#
# Links. You may link to our home page, provided you do so in a way that is fair and legal and does not damage our reputation or take advantage of it. You must not establish a link in such a way as to suggest any form of association, approval or endorsement on our part where none exists. You must not establish a link to our Services in any website that is not owned by you. The website in which you are linking must comply in all respects with the content standards set out at “Your Access to and Use of Our Services” above. We reserve the right to withdraw linking permission without notice.
#
# No Waiver. Our failure to insist upon or enforce any provision of these Terms shall not be construed as a waiver of any provision or right.
#
# Security. We do not guarantee that our Services will be secure or free from bugs or viruses. You are responsible for configuring your information technology, computer programmes and platform to access our Services. You should use your own virus protection software.
#
# Severability. If any court of law, having jurisdiction to decide on this matter, rules that any provision of these Terms is invalid, then that provision will be removed from the Terms without affecting the rest of the Terms, and the remaining provisions of the Terms will continue to be valid and enforceable.
#
# ARBITRATION AND CLASS ACTION WAIVER. This Section includes an arbitration agreement and an agreement that all claims will be brought only in an individual capacity (and not as a class action or other representative proceeding). Please read it carefully. You may opt out of the arbitration agreement by following the opt out procedure described below.
#
# Informal Process First. You agree that in the event of any dispute between you and TikTok, you will first contact TikTok and make a good faith sustained effort to resolve the dispute before resorting to more formal means of resolution, including without limitation any court action.
#
# Arbitration Agreement. After the informal dispute resolution process any remaining dispute, controversy, or claim (collectively, “Claim”) relating in any way to your use of TikTok’s services and/or products, including the Services, or relating in any way to the communications between you and TikTok or any other user of the Services, will be finally resolved by binding arbitration. This mandatory arbitration agreement applies equally to you and TikTok. However, this arbitration agreement does not (a) govern any Claim by TikTok for infringement of its intellectual property or access to the Services that is unauthorized or exceeds authorization granted in these Terms or (b) bar you from making use of applicable small claims court procedures in appropriate cases. If you are an individual you may opt out of this arbitration agreement within thirty (30) days of the first of the date you access or use this Services by following the procedure described below.
#
# You agree that the U.S. Federal Arbitration Act governs the interpretation and enforcement of this provision, and that you and TikTok are each waiving the right to a trial by jury or to participate in a class action. This arbitration provision will survive any termination of these Terms.
#
# If you wish to begin an arbitration proceeding, after following the informal dispute resolution procedure, you must send a letter requesting arbitration and describing your claim to:
#
# TikTok Inc., 5800 Bristol Parkway, Culver City, CA 90230
#
# Email Address: legal@tiktok.com
#
# The arbitration will be administered by the American Arbitration Association (AAA) under its rules including, if you are an individual, the AAA's Supplementary Procedures for Consumer-Related Disputes. If you are not an individual or have used the Services on behalf of an entity, the AAA's Supplementary Procedures for Consumer-Related Disputes will not be used. The AAA's rules are available at www.adr.org or by calling 1-800-778-7879.
#
# Payment of all filing, administration and arbitrator fees will be governed by the AAA's rules. If you are an individual and have not accessed or used the Services on behalf of an entity, we will reimburse those fees for claims where the amount in dispute is less than $10,000, unless the arbitrator determines the claims are frivolous, and we will not seek attorneys’ fees and costs in arbitration unless the arbitrator determines the claims are frivolous.
#
# The arbitrator, and not any federal, state, or local court, will have exclusive authority to resolve any dispute relating to the interpretation, applicability, unconscionability, arbitrability, enforceability, or formation of this arbitration agreement, including any claim that all or any part of this arbitration agreement is void or voidable. However, the preceding sentence will not apply to the “Class Action Waiver” section below.
#
# If you do not want to arbitrate disputes with TikTok and you are an individual, you may opt out of this arbitration agreement by sending an email to legal@tiktok.com within thirty (30) days of the first of the date you access or use the Services.
#
# Class Action Waiver. Any Claim must be brought in the respective party’s individual capacity, and not as a plaintiff or class member in any purported class, collective, representative, multiple plaintiff, or similar proceeding (“Class Action”). The parties expressly waive any ability to maintain any Class Action in any forum. If the Claim is subject to arbitration, the arbitrator will not have authority to combine or aggregate similar claims or conduct any Class Action nor make an award to any person or entity not a party to the arbitration. Any claim that all or part of this Class Action Waiver is unenforceable, unconscionable, void, or voidable may be determined only by a court of competent jurisdiction and not by an arbitrator. The parties understand that any right to litigate in court, to have a judge or jury decide their case, or to be a party to a class or representative action, is waived, and that any claims must be decided individually, through arbitration.
#
# If this class action waiver is found to be unenforceable, then the entirety of the Arbitration Agreement, if otherwise effective, will be null and void. The arbitrator may award declaratory or injunctive relief only in favor of the individual party seeking relief and only to the extent necessary to provide relief warranted by that party's individual claim. If for any reason a claim proceeds in court rather than in arbitration, you and TikTok each waive any right to a jury trial.
#
# If a counter-notice is received by TikTok’s Copyright Agent, we may send a copy of the counter-notice to the original complaining party informing that person that we may replace the removed content or cease disabling it. Unless the original complaining party files an action seeking a court order against the Content Provider, member or user, the removed content may be replaced, or access to it restored, in ten business days or more after receipt of the counter-notice, at TikTok’s sole discretion.
#
# Please understand that filing a counter-notification may lead to legal proceedings between you and the complaining party to determine ownership. Be aware that there may be adverse legal consequences in your country if you make a false or bad faith allegation by using this process.
#
# California Consumer Rights Notice. Under California Civil Code Section 1789.3, California users of the Services receive the following specific consumer rights notice: The Complaint Assistance Unit of the Division of Consumer Services of the California Department of Consumer Affairs may be contacted in writing at the contact information set forth at https://www.dca.ca.gov/about_us/contactus.shtml.
#
# Users of the Services who are California residents and are under 18 years of age may request and obtain removal of User Content they posted by contacting us at: https://www.tiktok.com/legal/report/feedback. All requests must be labeled "California Removal Request" on the email subject line. All requests must provide a description of the User Content you want removed and information reasonably sufficient to permit us to locate that User Content. We do not accept California Removal Requests via postal mail, telephone or facsimile. We are not responsible for notices that are not labeled or sent properly, and we may not be able to respond if you do not provide adequate information.
#
# Exports. You agree that you will not export or re-export, directly or indirectly the Services and/or other information or materials provided by TikTok hereunder, to any country for which the United States or any other relevant jurisdiction requires any export license or other governmental approval at the time of export without first obtaining such license or approval. In particular, but without limitation, the Services may not be exported or re-exported (a) into any U.S. embargoed countries or any country that has been designated by the U.S. Government as a “terrorist supporting” country, or (b) to anyone listed on any U.S. Government list of prohibited or restricted parties, including the U.S. Treasury Department's list of Specially Designated Nationals or the U.S. Department of Commerce Denied Person’s List or Entity List.
#
# U.S. Government Restricted Rights. The Services and related documentation are "Commercial Items", as that term is defined at 48 C.F.R. §2.101, consisting of "Commercial Computer Software" and "Commercial Computer Software Documentation", as such terms are used in 48 C.F.R. §12.212 or 48 C.F.R. §227.7202, as applicable. Consistent with 48 C.F.R. §12.212 or 48 C.F.R. §227.7202-1 through 227.7202-4, as applicable, the Commercial Computer Software and Commercial Computer Software Documentation are being licensed to U.S. Government end users (a) only as Commercial Items and (b) with only those rights as are granted to all other end users pursuant to the terms and conditions herein.
#
# App Stores
# To the extent permitted by applicable law, the following supplemental terms shall apply when accessing the Platform through specific devices:
#
# Notice regarding Apple.
# By downloading the Platform from a device made by Apple, Inc. (“Apple”) or from Apple’s App Store, you specifically acknowledge and agree that:
#
# These Terms between TikTok and you; Apple is not a party to these Terms.
# The license granted to you hereunder is limited to a personal, limited, non-exclusive, non-transferable right to install the Platform on the Apple device(s) authorised by Apple that you own or control for personal, non-commercial use, subject to the Usage Rules set forth in Apple’s App Store Terms of Services.
# Apple is not responsible for the Platform or the content thereof and has no obligation whatsoever to furnish any maintenance or support services with respect to the Platform.
# In the event of any failure of the Platform to conform to any applicable warranty, you may notify Apple, and Apple will refund the purchase price for the Platform, if any, to you. To the maximum extent permitted by applicable law, Apple will have no other warranty obligation whatsoever with respect to the Platform.
# Apple is not responsible for addressing any claims by you or a third party relating to the Platform or your possession or use of the Platform, including without limitation (a) product liability claims; (b) any claim that the Platform fails to conform to any applicable legal or regulatory requirement; and (c) claims arising under consumer protection or similar legislation.
# In the event of any third party claim that the Platform or your possession and use of the Platform infringes such third party’s intellectual property rights, Apple is not responsible for the investigation, defence, settlement or discharge of such intellectual property infringement claim.
# You represent and warrant that (a) you are not located in a country that is subject to a U.S. Government embargo, or that has been designated by the U.S. Government as a “terrorist supporting” country; and (b) you are not listed on any U.S. Government list of prohibited or restricted parties.
# Apple and its subsidiaries are third party beneficiaries of these Terms and upon your acceptance of the terms and conditions of these Terms, Apple will have the right (and will be deemed to have accepted the right) to enforce these Terms against you as a third party beneficiary hereof.
# TikTok expressly authorises use of the Platform by multiple users through the Family Sharing or any similar functionality provided by Apple.
# Windows Phone Store.
# By downloading the Platform from the Windows Phone Store (or its successors) operated by Microsoft, Inc. or its affiliates, you specifically acknowledge and agree that:
#
# You may install and use one copy of the Platform on up to five (5) Windows Phone enabled devices that are affiliated with the Microsoft account you use to access the Windows Phone Store. Beyond that, we reserve the right to apply additional conditions or charge additional fees.
# You acknowledge that Microsoft Corporation, your phone manufacturer and network operator have no obligation whatsoever to furnish any maintenance and support services with respect to the Platform.
# Amazon Appstore.
# By downloading the Platform from the Amazon Appstore (or its successors) operated by Amazon Digital Services, Inc. or affiliates (“Amazon”), you specifically acknowledge and agree that:
#
# to the extent of any conflict between (a) the Amazon Appstore Terms of Use or such other terms which Amazon designates as default end user license terms for the Amazon Appstore (“Amazon Appstore EULA Terms”), and (b) the other terms and conditions in these Terms, the Amazon Appstore EULA Terms shall apply with respect to your use of the Platform that you download from the Amazon Appstore, and
# Amazon does not have any responsibility or liability related to compliance or non-compliance by TikTok or you (or any other user) under these Terms or the Amazon Appstore EULA Terms.
# Google Play.
# By downloading the Platform from Google Play (or its successors) operated by Google, Inc. or one of its affiliates (“Google”), you specifically acknowledge and agree that:
#
# to the extent of any conflict between (a) the Google Play Terms of Services and the Google Play Business and Program Policies or such other terms which Google designates as default end user license terms for Google Play (all of which together are referred to as the “Google Play Terms”), and (b) the other terms and conditions in these Terms, the Google Play Terms shall apply with respect to your use of the Platform that you download from Google Play, and
# you hereby acknowledge that Google does not have any responsibility or liability related to compliance or non-compliance by TikTok or you (or any other user) under these Terms or the Google Play Terms.
# Contact Us.
# You can reach us at: https://www.tiktok.com/legal/report/feedback or write us at TikTok Inc.: 10100 Venice Blvd., Culver City, CA 90232 , USA
#
#
# Company
# About TikTok
# Newsroom
# Contact
# Careers
# ByteDance
# Programs
# TikTok for Good
# TikTok for Developers
# Effect House
# Advertise on TikTok
# TikTok Browse
# TikTok Embeds
# TikTok Rewards
# Resources
# Help Center
# Safety Center
# Creator Portal
# Community Guidelines
# Transparency
# Accessibility
# Legal
# Terms of Service
# Privacy Policy
# Children's Privacy Policy
# TikTok Platform Cookies Policy
# Intellectual Property Policy
# Law Enforcement Guidelines
# © 2023 TikTok"""
#
#     print(get_results(text=text))
