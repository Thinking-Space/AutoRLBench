/* People shown on the Team page, in display order.
   name:        display name
   handle:      GitHub username, used only for the avatar (github.com/<handle>.png);
                leave empty to show initials instead
   affiliation: one line, e.g. "University · Lab" */
const TEAM = [
    { name: 'Joy Lim*', handle: 'JoylimJY', affiliation: 'Tsinghua University' },
    { name: 'Yinghao Chen*', handle: 'yhyayi', affiliation: 'Tsinghua University' },
    { name: 'Bingxiang He*', handle: 'HBX-hbx', affiliation: 'Tsinghua University' },
    { name: 'Yuxin Zuo*', handle: 'yuxinzuo', affiliation: 'Tsinghua University' },
    { name: 'Yaohui Li', handle: 'lif1014', affiliation: 'Tsinghua University' },
    { name: 'Jiarui Yuan', handle: '1rubbishyuan', affiliation: 'Tsinghua University' },
    { name: 'Zixuan Fu', handle: 'Zixuan-Fu', affiliation: 'Tsinghua University' },
    { name: 'Yuan Gao', handle: 'Zaogezi', affiliation: 'Tsinghua University' },
    { name: 'Xin Huang', handle: 'NaCresp', affiliation: 'Tsinghua University' },
    { name: 'Yaxuan Li', handle: 'YaxuanLi-cn', affiliation: 'Tsinghua University' },
    { name: 'Zixi Chen', handle: 'Jccc1025', affiliation: 'Tsinghua University' },
    { name: 'Shouzheng Huang', handle: 'Huangsz2021', affiliation: 'Harbin Institute of Technology (Shenzhen)' },
    { name: 'Jinyang Wu', handle: 'jinyangwu', affiliation: 'Tsinghua University' },
    { name: 'Junlin Yang', handle: 'YangJL2003', affiliation: 'Tsinghua University' },
    { name: 'Jianing Yin', handle: 'CatherineYin1213', affiliation: 'University of Pennsylvania'},
    { name: 'Ruijia Zhang', handle: 'Richard-ZZZ', affiliation: 'Johns Hopkins University'},
    { name: 'Haoyang Su', handle: 'Hoyant-Su', affiliation: 'Fudan University'},
    { name: 'Pengyi Li', handle: 'lipengyi0829', affiliation: 'FusionBrain Lab'},
    { name: 'Purva Chiniya', handle: 'PurvaChiniya', affiliation: 'Independent Researcher'},
    { name: 'Dengzhe Hou', handle: 'dengzhe-hou', affiliation: 'Tohoku University'},
    { name: 'Ziqi Zhou', handle: 'ziqi-zhou02', affiliation: 'Duke University'}
];

/* The institution wall below the team grid is derived from the affiliations
   above: unique names, in order of first appearance. Multi-part
   affiliations ("University · Lab") contribute each part once.
   Affiliations that are not institutions are listed here so they stay
   off the wall. */
const NON_INSTITUTIONS = ['Independent Researcher'];
const INSTITUTIONS = Array.from(new Set(TEAM.flatMap(person =>
    String(person.affiliation || '')
        .split(/[·;]/)
        .map(part => part.trim())
))).filter(part => part && !NON_INSTITUTIONS.includes(part));

/* Labs and programs shown on the wall alongside the universities above.
   Each entry is spliced in after the named institution; the logo map
   below assigns them a cell like any other entry. */
const INSTITUTION_INSERTS = [
    { after: 'Tsinghua University', name: 'THUNLP' }
];

/* The logos on the Team page all live in one image, images/institutions.png:
   a single row of 9 cells (120x60 each, rendered at 2x -> 2160x120).
   INSTITUTION_LOGOS maps each institution to its cell; institutions without
   an entry render name-only. To add an institution, drop its logo file into
   page/scripts/institutions/ (untracked), extend compose-strip.html there,
   and re-render it headless at window-size 2160x120. */
const INSTITUTION_LOGOS = {
    'Tsinghua University': 0,
    'THUNLP': 1,
    'Harbin Institute of Technology (Shenzhen)': 2,
    'University of Pennsylvania': 3,
    'Johns Hopkins University': 4,
    'Fudan University': 5,
    'FusionBrain Lab': 6,
    'Tohoku University': 7,
    'Duke University': 8
};
