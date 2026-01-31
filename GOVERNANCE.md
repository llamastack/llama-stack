# Llama Stack Governance

## Introduction

Llama Stack is an open-source project that standardizes the core building blocks that simplify AI application
development, it is maintained by a community of developers and contributors.
This document outlines the governance model for the Llama Stack project.

The Llama Stack project aims for open and transparent governance and decision-making, thus encouraging community
building and contribution.

A formal governance structure helps us to
*   Provide a structure for individuals to become involved in the project.
*   Communicate all processes for members to operate within the project.
*   Document a system for open product development, roadmapping, and planning.
*   Provide a means for making decisions if consensus cannot be reached.

## Community Overview

At a high level, the key moving parts of the community are:
- **GitHub activity** (issues + pull requests + discussions)

With this structure, users and contributors largely self-organize and contribute changes as per [lazy consensus](#lazy-consensus). If there is active opposition and unresolvable conflict, then Code Owners step in to break ties or make decisions.

## Llama Stack Governance Model
Llama Stack is a meritocratic, consensus-based community project.

Anyone interested in the project can join the community to:
- contribute to the project design
- participate in the decision-making process.

> **Note**: There may not always a corresponding CODEOWNER for the affected code, in which case the responsibility falls
> on other Code Owners or contributors with write access to review and merge the PR.

## Roles and Responsibilities

There are several roles within the Llama Stack community, each with its own set of responsibilities. These roles are
- [Users](#users)
- [Contributors](#contributors)
- [Triagers](#triagers)
- [Code Owners](#code-owners)

### Users

Users are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and are encouraged to participate in the
community.

Users are community members who require the operational functionality of Llama Stack. They are the most important
community members, and without them, the project would have no purpose. Anyone can be a user; there are no special
requirements.

Llama Stack asks its users to participate in the project and community as much as possible. User contributions enable
the project team to ensure that they satisfy the needs of those users. Frequently, user contributions include (but are
not limited to):
* Providing developers with feedback on the project (i.e., the user experience)
* Providing feature requests
* Filing bug reports or flagging issues
* Providing moral support
* Evangelizing the project (e.g., On Social Media)

Users who continue to engage with the project and its community will often become more and more involved. Such users may
find themselves becoming contributors, as described in the next section.

### Contributors

Contributors are community members who contribute in concrete ways to the project. Anyone can become a contributor, and
contributions can take many forms, as detailed in the [all-contributors project](https://allcontributors.org/docs/en/emoji-key#table).
There is no expectation of commitment to the project, no specific skill requirements, and no selection process.

In addition to their actions as users, contributors may also find themselves doing one or more of the following:
*   Supporting new users (existing users are often the best people to help new users)
*   Creating, triaging or commenting on GitHub Issues
*   Doing code reviews or commenting on technical documents
*   Writing, editing, translating or reviewing the documentation
*   Organizing events or evangelizing the project (e.g,. on Social Media)

Contributors engage with the project through GitHub issues or by writing or editing documentation. They submit changes to the project itself via Pull Requests (PRs), which will be considered for inclusion in the project by existing Code Owners (see next section).

Contributors should follow the following guides when creating PRs:
- [Contribution Process](CONTRIBUTING.md)

As contributors gain experience and familiarity with the project, their profile and commitment within the community will
increase. At some stage, they may find themselves being nominated to become a Triager or Code Owner by an
existing Code Owner.

### Triagers
Triagers (mapped in [TRIAGERS.md](./.github/TRIAGERS.md)) are contributors who have shown that they are committed to Llama Stack's continued development by helping to
manage (track, label, assign issues - general issue hygiene) the project's backlog. They are responsible for ensuring that the project's issues are well-organized and up to
date so that contributors and Code Owners can navigate the project efficiently and prioritize key requests from
the community.

#### Nomination Process

Any Code Owner may nominate a contributor to become a Triager by opening a PR that adds their name
to [TRIAGERS.md](./.github/TRIAGERS.md).

The nomination follows **lazy consensus**:
- If no Code Owner objects within 72 hours, the PR is merged.
- If concerns are raised, they should be discussed constructively in the PR.
- If consensus cannot be reached, maintainers will make the final decision.

Once merged, the contributor is granted the GitHub `Triage` role.

### Code Owners

Code Owners (mapped in [CODEOWNERS](./.github/CODEOWNERS)) are community members who have shown that they are committed to Llama Stack's continued development through
ongoing engagement with the community. Because of this, they have the right to merge PRs and have voting rights.

> **Note**: Code Owners, like other contributors, must make changes to Llama Stack via pull requests (with code review).
> This applies to all changes to documentation, code, configuration, governance.

#### Code Owner Responsibilities
Code Owners are responsible for reviewing and merging PRs for the projects and areas they are
assigned to. They also
- participate in strategic planning, approve changes to the governance model, and manage the
  copyrights within the project outputs.
- triage GitHub issues, applying [labels](https://github.com/llamastack/llama-stack/labels) to each new item. Labels are extremely useful for future issue follow ups. Adding labels is somewhat subjective, so please use your best judgment.
- make sure that ongoing PRs are moving forward at the right pace or closing them.


### Becoming a Code Owner

#### Who is eligible to become a Code Owner?
Anyone can become a Code Owner. Typically, a potential Code Owner will need to show that they understand the project,
its objectives, and its strategy. They will also have provided valuable contributions to the project over a period of
time. Code Owners must also act in the interest of the community.

#### Process
Any existing Code Owner may nominate a contributor to join the Code Owners group.

Code Owner nominations and voting take place **privately** among the current Code Owners.
This privacy ensures that Code Owners can share candid assessments without risking public
embarrassment for the nominee.

A nomination is approved when it receives a **majority vote (>50%)** from the current
Code Owners.

If a nomination is not approved, the nominee should receive constructive feedback so they
understand the reasons and can continue to grow toward the role.


#### Earning a Nomination

The nomination is at the discretion of the existing Code Owners and there is no single path of
earning a nomination. You don't have to be a triager to become a Code Owner, however, we can give some guidance about some actions that would help:

*   Start by expressing interest to the Code Owners that you are interested in becoming a Code Owner.
*   You can start tackling issues labeled as ‘help wanted’, or if you are new to the project, some of the ‘good first issue’ tickets. As you gain experience with the codebase and our standards, we will ask you to do code reviews for incoming PRs (i.e., all Code Owners are expected to shoulder a proportional share of community reviews).
*   We will expect you to start contributing increasingly complicated PRs, under the guidance of the existing Code Owners.

#### Losing Code Owner Status

If a Code Owner is no longer interested and cannot perform the Code Owner duties listed above, they can volunteer to be
moved to emeritus status. The Code Owner status is attributed for life otherwise. An emeritus Code Owner may request
reinstatement of commit access from the rest of Code Owners. Such reinstatement is subject to lazy consensus approval of
active Code Owners.

Emeritus status is a nominal title, and confers no special rights (like voting) or access. Emeritus
members are functionally identical to normal contributors, with the exception that they can request
for reinstatement of their CODEOWNER access.

In extreme cases, Code Owners can lose their status by a vote of the Code Owners per the voting process below.

## Decision Making Process

Decisions about the future of Llama Stack are made through discussion with all community members, from the newest user
to the most experienced Code Owner. All non-sensitive project management discussion takes place on GitHub issues and discussions. Occasionally, sensitive discussion occurs on a private channel of our Discord.

To ensure that the project is not bogged down by endless discussion and continual voting, the project operates a policy
of lazy consensus. This allows the majority of decisions to be made without resorting to a formal vote.


### Lazy consensus

Decision making typically involves the following steps:
*   Discussion *(in Discord channels and GitHub)*
*   Proposal *(via GitHub issue + GitHub PR)*
*   (optional) Code Owners voting (if there is active opposition + consensus is not reached through discussion)
*   Decision

Any community member can make a proposal for consideration by the community. To initiate a discussion about a new idea,
they should create an issue or submit a PR implementing the idea. This will prompt a review and,
if necessary, a discussion of the idea. The goal of this review and discussion is to gain approval for the contribution.
Since most people in the project community have a shared vision, there is often little discussion to reach consensus.

In general, as long as nobody explicitly opposes a proposal or PR, it is recognized as having the support of the
community. This is called lazy consensus - that is, those who have not stated their opinion explicitly have implicitly
agreed to the proposal's implementation.

Lazy consensus is a fundamental concept within the project. This process allows a large group of people to reach
consensus efficiently as someone with no objections to a proposal need not spend time stating their position.

For lazy consensus to be effective, it is necessary to allow at least 48 hours (excluding weekends) before assuming that
there are no objections to the proposal. This requirement ensures that everyone is given enough time to read, digest,
and respond to the proposal. This time period is chosen to be as inclusive as possible of all participants, regardless
of their location and time commitments.

### Voting

Not all decisions can be made using lazy consensus. Issues such as those affecting the strategic direction or legal
standing of the project must gain explicit approval in the form of a vote. Every member of the community is encouraged
to express their opinions in all discussions and all votes. However, only project Code Owners have binding votes for
the purposes of decision making.


### Changes to Governance

We believe governance needs to adapt in order to be effective long term. This governance document itself can be extended
or modified as our community and project grows and our needs change.

A change in our governance structure should be a rare occurrence and should face sufficient scrutiny and review. To
this end, the rules that apply to modifications to the Llama Stack Governance structure are more stringent:

*   Governance changes are made through PRs to the repository.
*   Lazy consensus applies to governance changes, but the proposed changes must be public for at least 7 days instead of 48 hours before they are accepted.
*   If there is opposition to a change, a vote will be held by Code Owners.
*   Voting is asynchronous. All Code Owners must be notified of a vote through the Code Owner mailing list.
*   Code Owners must be given at least 7 days to respond.
*   Voting requires a super-majority in order to pass a decision, and Code Owners do not hold veto power for these votes. A super-majority is defined as at least 60% of votes in favor.
*   The total pool of votes does not include those who abstain from voting.
*   A quorum is required for voting. A quorum is 60% of Code Owners.


### Roadmap Creation

Our public roadmap gives an overview of what we are currently working on and what we want to tackle next. This
helps potential contributors understand our project's current status and where it's going next, as well as giving a
chance to be part of the planning.

In this section, we describe the process we follow to create it, using regular GitHub issues from
the Feature Request template.

### New Feature Process

Most of the issues we see can be handled with regular GitHub issues. However, some changes are "substantial", and we
ask that these go through a design process and produce a consensus among the Llama Stack community and Code Owners.

1. Contributor creates an issue using the Feature Request template
2. Users, Contributors, Triagers, and Code Owners discuss and upvote the issue
3. Code Owners approve issue when it is ready

If you submit a pull request to implement a new feature without going through the RFC process, it may be closed with a
polite request to submit an RFC first. That said, if most of the work is done, we'd accelerate the process.

## Resources

*   [Envoy’s Governance Document](https://github.com/envoyproxy/envoy/blob/master/GOVERNANCE.md)
*   [OSS Watch, Meritocratic Governance](http://oss-watch.ac.uk/resources/meritocraticgovernancemodel)
*   [The Apache Software Foundation meritocratic model](http://www.apache.org/foundation/how-it-works.html#meritocracy)
*   [Ember RFCs](https://github.com/emberjs/rfcs)

Attribution: The Llama Stack governance structure is based on the Amundsen and Feast Governance
structure.
