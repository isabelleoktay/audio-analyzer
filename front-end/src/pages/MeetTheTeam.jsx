// import SurveyTextAnswer from "../components/survey/SurveyTextAnswer";
// import SecondaryButton from "../components/buttons/SecondaryButton";
import TeamMemberCard from "../components/cards/TeamMemberCard";

const MeetTheTeam = () => {
  return (
    <div className="min-h-screen justify-center pt-20 flex flex-col">
      {/* TEAM */}
      <div className="w-full flex justify-center p-10">
        <div className="w-full flex flex-col bg-bluegray/25 rounded-3xl p-10 gap-8">
          <h1 className="text-4xl font-bold text-lightpink text-center w-full">
            The MuSA Development Team
          </h1>
          <div className="w-full flex p-10 gap-8">
            {/* Team Member Cards */}
            <TeamMemberCard
              name="Isabelle Oktay"
              title=""
              image="/images/isabelle.jpeg"
              bio="Isabelle has a Master in Sound and Music Computing from Universitat Pompeu Fabra, Barcelona and a Bachelors in Computer Science from NYU, New York. The MuSA performance analyzer was her masters thesis project at UPF
            and she continues to work with the MuSA team as an independent developer."
              email=" "
              links={[
                {
                  type: "linkedin",
                  url: "https://www.linkedin.com/in/isabelle-oktay/",
                },
                { type: "github", url: "https://github.com/isabelleoktay/" },
              ]}
            ></TeamMemberCard>

            <TeamMemberCard
              name="Suvi Häärä"
              title="Predoctoral Researcher"
              image="/images/suvi.jpeg"
              bio="Suvi is completing her PhD in ICT at Universitat Pompeu Fabra, developing intelligent systems for singing pedagogy. She has a Master in Sound and Music Computing from UPF and a 
            Bachelors in Electronic and Electrical Engineering from Queen Mary University of London."
              email="suvi.haeaerae@upf.edu"
              links={[
                {
                  type: "linkedin",
                  url: "https://www.linkedin.com/in/suvi-h%C3%A4%C3%A4r%C3%A4-333127146/",
                },
                { type: "github", url: "https://github.com/suvimh/" },
                {
                  type: "orcid",
                  url: "https://orcid.org/0009-0009-1236-3873",
                },
              ]}
            ></TeamMemberCard>

            <TeamMemberCard
              name="Dr. Rafael Ramirez-Melendez"
              title="Tenured Associate Professor and Head of the Music and Machine Learning Lab at Universitat Pompeu Fabra"
              image="/images/rafael.jpeg"
              bio="Dr. Rafael Ramirez is the PhD supervisor of Suvi Häärä, overseeing the development of MuSA Voice. "
              email="rafael.ramirez@upf.edu"
              links={[
                {
                  type: "linkedin",
                  url: "https://www.linkedin.com/in/rafaelr2/",
                },
                {
                  type: "website",
                  url: "https://www.upf.edu/web/rafael-ramirez",
                },
                {
                  type: "orcid",
                  url: "https://orcid.org/0000-0002-3294-0764",
                },
                {
                  type: "g_scholar",
                  url: "https://scholar.google.com/citations?user=y-Qfon8AAAAJ&hl=en&oi=ao",
                },
              ]}
            ></TeamMemberCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MeetTheTeam;
