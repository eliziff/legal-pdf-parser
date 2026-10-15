use legal_structure_model::DetectionProfile;

pub const NAMES: &[&str] = &[
    "profile_known",
    "profile_case_rooted_complete",
    "profile_case_contiguous_complete",
    "profile_case_lossy",
    "profile_legislation",
    "profile_instrument",
    "profile_journal",
];

pub fn values(profile: Option<DetectionProfile>) -> [f32; 7] {
    let mut values = [0.0; 7];
    if let Some(profile) = profile {
        values[0] = 1.0;
        values[match profile {
            DetectionProfile::CaseRootedComplete => 1,
            DetectionProfile::CaseContiguousComplete => 2,
            DetectionProfile::CaseLossy => 3,
            DetectionProfile::Legislation => 4,
            DetectionProfile::Instrument => 5,
            DetectionProfile::Journal => 6,
        }] = 1.0;
    }
    values
}
