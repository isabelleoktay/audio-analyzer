import React from "react";
import {
  FaLinkedin,
  FaGithub,
  FaEnvelope,
  FaExternalLinkAlt,
  FaGlobe,
  FaOrcid,
} from "react-icons/fa";
import { FaGoogleScholar } from "react-icons/fa6";

const TeamMemberCard = ({ name, title, bio, email, image, links = [] }) => {
  /**
   * links prop format:
   * [
   *   { type: 'linkedin', url: 'https://linkedin.com/...' },
   *   { type: 'email', url: 'mailto:...' },
   *   { type: 'website', url: 'https://...' },
   *   { type: 'github', url: 'https://github.com/...' }
   * ]
   */

  const getIconForType = (type) => {
    switch (type.toLowerCase()) {
      case "linkedin":
        return <FaLinkedin size={18} />;
      case "email":
        return <FaEnvelope size={18} />;
      case "g_scholar":
        return <FaGoogleScholar size={18} />;
      case "github":
        return <FaGithub size={18} />;
      case "website":
        return <FaGlobe size={18} />;
      case "orcid":
        return <FaOrcid size={18} />;
      default:
        return <FaExternalLinkAlt size={18} />;
    }
  };

  return (
    <div className="w-full max-w-sm mx-auto">
      {/* Card Container */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl overflow-hidden shadow-xl hover:shadow-2xl transition-shadow duration-300 border border-slate-700/50">
        {/* Image Container */}
        <div className="relative overflow-hidden bg-slate-700 aspect-square">
          {image ? (
            <img
              src={image}
              alt={name}
              className="w-full h-full object-cover transition-transform duration-500 hover:scale-105"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-600 to-slate-800">
              <span className="text-slate-400 text-sm">No image</span>
            </div>
          )}
        </div>

        {/* Content Container */}
        <div className="p-6 space-y-4">
          {/* Name */}
          <div>
            <h3 className="text-2xl font-bold text-white tracking-tight">
              {name}
            </h3>
          </div>

          {/* Title */}
          {title && (
            <p className="text-pink-300 text-sm font-semibold uppercase tracking-wider">
              {title}
            </p>
          )}

          {/* Bio/Details */}
          {bio && (
            <p className="text-gray-300 text-sm leading-relaxed">{bio}</p>
          )}

          {/* Email */}
          {email && (
            <a
              href={`mailto:${email}`}
              className="block text-pink-200 text-xs hover:text-pink-100 transition-colors break-all"
            >
              {email}
            </a>
          )}

          {/* Social Links */}
          {links.length > 0 && (
            <div className="flex gap-3 pt-2">
              {links.map((link, index) => (
                <a
                  key={index}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center w-10 h-10 rounded-lg bg-pink-500/20 text-pink-300 hover:bg-pink-500/40 hover:text-pink-100 transition-all duration-200 hover:scale-110"
                  title={link.type}
                  aria-label={`${link.type} link`}
                >
                  {getIconForType(link.type)}
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TeamMemberCard;
