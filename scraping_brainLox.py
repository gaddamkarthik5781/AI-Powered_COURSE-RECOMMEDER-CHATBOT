import requests
from bs4 import BeautifulSoup
import json

base_url = "https://brainlox.com"
url = "https://brainlox.com/courses/category/technical"
response = requests.get(url)
print(response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

courses = []
for course in soup.find_all("div", class_="courses-content"):  
    if course.find("h3"):
        title = course.find("h3").get_text(strip=True)
    else:
        title = "No Title"

    link_tag = course.find("a")
    if link_tag:
        course_url = base_url + link_tag.get('href')

        course_response = requests.get(course_url)
        course_soup = BeautifulSoup(course_response.text, "html.parser")

        detailed_description = course_soup.find('div', class_="courses-overview")
        if detailed_description and detailed_description.find("p"):
            description = detailed_description.find("p").get_text(separator=" ").strip()
            description = "\n".join([line.strip() for line in description.splitlines() if line.strip()])
            description = description.replace("MAIN FEATURES OF THE PROGRAM", "\nMAIN FEATURES OF THE PROGRAM : \n")
        else:
            description = "No Description"

        # Extract Enquire and Demo Booking links
        enquire_link = ""
        demo_booking_link = ""

        btn_box = course_soup.find('div', class_='btn-box')
        if btn_box:
            all_links = btn_box.find_all('a')
            unique_hrefs = set()
            for a in all_links:
                href = a.get('href')
                if href:
                    unique_hrefs.add(href)

            for link in unique_hrefs:
                if 'contact' in link:
                    enquiry_link = base_url + link
                elif 'book-free-demo' in link:
                    demo_booking_link = base_url + link

        courses.append({
            "Title": title,
            "Description": description,
            "More_Details_Link": course_url,
            "Curriculum_Link": course_url,
            "Course_Fee_Link": course_url,
            "Enquire_Link": enquiry_link,
            "Demo_Booking_Link": demo_booking_link,
            "Note": "Curriculum is visible on the same page. Please click the 'Curriculum' tab manually to view it."
        })


with open("courses.json", "w", encoding="utf-8") as file:
    json.dump(courses, file, ensure_ascii=False, indent=4)

print("Courses Fetched and saved succesfully...!")